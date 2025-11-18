import random
from datetime import datetime
from typing import Any
import os
import dotenv
import lightning as L
import numpy as np
import torch
from cirkit.backend.torch.queries import SamplingQuery
from cirkit.pipeline import compile as cirkit_compile
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.utils import (
    GreenCallback,
    StatusExceptionCallback,
    circuit_factory,
    new_csv_logger,
    patchify,
    write_summary_yaml,
)

dotenv.load_dotenv()


# define the LightningModule
class BenchPCImage(L.LightningModule):
    def __init__(self, config: dict[str, Any], len_dataset_test: int):
        super().__init__()
        self.key, self.symbolic_circuit = circuit_factory(**config)
        self.circuit = cirkit_compile(self.symbolic_circuit)
        self.config = config

        self.circuit_type = self.config["circuit_type"]
        self.layer_type = self.config["layer_type"]
        self.region_graph = self.config["region_graph"]
        self.num_units = self.config["num_units"]
        self.lr = self.config["lr"]
        self.kernel_size = self.config["kernel_size"]
        self.image_size = self.config["image_size"]
        self.channel = self.config["channel"]
        self.patch_fn = patchify(self.kernel_size, self.kernel_size, compile=False)
        self.patch_per_img = (self.image_size[0] // self.kernel_size[0]) * (
            self.image_size[1] // self.kernel_size[1]
        )
        self.len_dataset_test = len_dataset_test
        self.save_hyperparameters()

    def on_training_start(self):
        torch.cuda.memory.reset_peak_memory_stats()

    def training_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)

        log_likelihoods = self.circuit(batch)
        loss = -torch.mean(log_likelihoods)

        # Log directly to Comet offline
        if self.circuit_type == "patch":
            self.log("train_loss", (loss * self.patch_per_img).item())
        else:
            self.log("train_loss", loss.item())
        return loss

    def on_training_end(self):
        self.log(
            "max_train_memory", torch.cuda.memory.max_memory_allocated(self.device)
        )

    def on_validation_epoch_start(self):
        self.running_loss = []

    def validation_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)

        # Compute the log-likelihoods of the batch, by evaluating the circuit
        log_likelihoods = self.circuit(batch)

        # We take the negated average log-likelihood as loss
        loss = -torch.mean(log_likelihoods).item()
        if self.circuit_type == "patch":
            self.running_loss.append(loss * self.patch_per_img)
        else:
            self.running_loss.append(loss)
        # Logging to TensorBoard (if installed) by default

    def on_validation_epoch_end(self):
        val_loss = sum(self.running_loss) / len(self.running_loss)
        self.log("val_loss", val_loss)

    def on_test_epoch_start(self):
        self.test_loss = 0

    def prepare_batch(self, batch):
        batch, _ = batch

        # this is the test loop
        if self.circuit_type == "patch":
            batch = self.patch_fn(batch)
        BS = batch.shape[0]
        batch = batch.view(BS, -1)
        return batch

    def test_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        log_likelihoods = self.circuit(batch)
        test_lls = -log_likelihoods.sum().item()
        self.test_loss += test_lls

    def on_test_epoch_end(self):
        average_nll = self.test_loss / self.len_dataset_test
        self.log("test_loss", average_nll)
        self.log("test_bpd", self.get_bpd(average_nll))

        if self.layer_type != "tucker":
            query = SamplingQuery(self.circuit)
            num_samples = 10
            if self.circuit_type == "patch":
                num_samples *= self.patch_per_img
                print(f"Num patch to sample : {num_samples}")

            samples, _ = query(num_samples=num_samples)
            img_samples = (
                samples.reshape((10, self.channel, *self.image_size)).float().to("cpu")
            )  # (N, C, H, W)
            self.logger.log_samples(img_samples)

    def get_bpd(self, average_nll: float):
        img_dim = self.channel * self.image_size[0] * self.image_size[1]
        return average_nll / (img_dim * np.log(2.0))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def benchmark(config, train_dataloader, val_dataloader, test_dataloader, turn_off_logs=False):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    light_mode = BenchPCImage(config, len(test_dataloader.dataset))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stop = EarlyStopping(monitor="val_loss", mode="min")
    exception_callback = StatusExceptionCallback()
    green_callback = GreenCallback()
    logger = new_csv_logger(light_mode.key, exp_dir=config["experiment_path"])

    log_conf = {}
    if turn_off_logs:
        log_conf["enable_progress_bar"]=False
        log_conf["enable_model_summary"]=False

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        logger=logger,
        max_epochs=300,
        callbacks=[early_stop, checkpoint_callback, exception_callback, green_callback],
        num_sanity_val_steps=10,
        log_every_n_steps=10,
        inference_mode=False,
        **log_conf
    )

    try:
        trainer.fit(
            light_mode,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        light_mode = BenchPCImage.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        test_results = trainer.test(
            light_mode,
            dataloaders=test_dataloader,
        )
        if test_results:
            test_metrics = test_results[0]  # first dict in list
        else:
            test_metrics = {}

        # Return config + metrics
        return {
            "config": config,
            "test_metrics": test_metrics,
        }

    except Exception as e:
        logger.summary["exception"] = str(e)
        logger.summary["end_datetime"] = datetime.now().isoformat()
        logger.summary["status"] = "FAILED"
        write_summary_yaml(logger.summary_path, logger.summary)
        raise


def benchmark_dataset(config):

    if config["dataset"] == "mnist":
        dataloaders = load_mnist(config)
    elif config["dataset"] == "cifar":
        dataloaders = load_cifar(config)
    else:
        print(f"Error: no dataset named {config['dataset']}")
        return
    turn_off_logs = False
    if os.getenv("IN_RAY_TASK") == "1":
        turn_off_logs=True
    return benchmark(config, **dataloaders, turn_off_logs=turn_off_logs)


def load_mnist(config):
    if "batch_size" in config:
        batch_size = config["batch_size"]
    else:
        batch_size = 64
    config.update(
        {
            "image_size": (28, 28),
            "channel": 1,
        }
    )
    pixel_range = 255
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (pixel_range * x).long()),
        ]
    )

    data_train = datasets.MNIST(
        "datasets", train=True, download=True, transform=transform
    )
    data_test = datasets.MNIST(
        "datasets", train=False, download=True, transform=transform
    )

    train_idx, val_idx = train_test_split(range(len(data_train)), test_size=0.25)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    # Instantiate the training and testing data loaders
    train_dataloader = DataLoader(
        data_train, batch_size=batch_size, sampler=train_sampler
    )
    val_dataloader = DataLoader(data_train, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(data_test, shuffle=False, batch_size=batch_size)

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
    }


def load_cifar(config):
    if "batch_size" in config:
        batch_size = config["batch_size"]
    else:
        batch_size = 64
    config.update(
        {
            "image_size": (32, 32),
            "channel": 3,
        }
    )
    pixel_range = 255
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (pixel_range * x).long()),
        ]
    )

    data_train = datasets.CIFAR10(
        "datasets", train=True, download=True, transform=transform
    )
    data_test = datasets.CIFAR10(
        "datasets", train=False, download=True, transform=transform
    )

    train_idx, val_idx = train_test_split(range(len(data_train)), test_size=0.25)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    # Instantiate the training and testing data loaders
    train_dataloader = DataLoader(
        data_train, batch_size=batch_size, sampler=train_sampler
    )
    val_dataloader = DataLoader(data_train, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(data_test, shuffle=False, batch_size=batch_size)

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
    }
