import os
import random
import gc
from datetime import datetime
from typing import Any

import dotenv
import lightning as L
import numpy as np
import torch
from cirkit.backend.torch.parameters.pic import pc2qpc
from cirkit.backend.torch.queries import SamplingQuery
from cirkit.pipeline import PipelineContext
from cirkit.pipeline import compile as cirkit_compile
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import optim

from src.colours import Rgb2GenT
from src.config import DATASET_ROOT
from src.dataset import CIFAR, MNIST, CelebA, ImageNet
from src.utils import (
    GreenCallback,
    StatusExceptionCallback,
    circuit_factory,
    new_csv_logger,
    patchify,
    unpatchify,
    write_summary_yaml,
)
from src.share_param import get_composite_circ, get_share_circ

dotenv.load_dotenv()


# define the LightningModule
class BenchPCImage(L.LightningModule):
    def __init__(self, config: dict[str, Any], load_composite=False):
        super().__init__()
        # print("initializing benchmark PC", flush=True)
        self.key, self.symbolic_circuit = circuit_factory(**config)
        self.use_pic = config.get("use_pic", False)

        if config["circuit_type"] == "composite":
            assert "patch_ckpt" in config, "Missing path to patch checkpoint"
            patch_module = BenchPCImage.load_from_checkpoint(
                config["patch_ckpt"], map_location="cpu", load_composite=True
            )
            # print("getting composite circuit", flush=True)

            self.circuit = get_composite_circ(config, patch_module)
            # print("composite circuit built", flush=True)

            self.kernel_size = patch_module.config["kernel_size"]
        elif config["circuit_type"] == "shared":
            self.circuit = get_share_circ(config)
            self.kernel_size = config["kernel_size"]

        elif self.use_pic:
            self.circuit = cirkit_compile(
                self.symbolic_circuit,
                PipelineContext(semiring="lse-sum", fold=True, optimize=False),
            )
            print("GET QPC PIC")
            pc2qpc(self.circuit, integration_method="trapezoidal", net_dim=256)
            self.kernel_size = config["kernel_size"]
        else:
            self.circuit = cirkit_compile(self.symbolic_circuit)
            self.kernel_size = config["kernel_size"]
        # print("benchmark PC initialized", flush=True)

        self.config = config
        self.circuit_type = self.config["circuit_type"]
        self.layer_type = self.config["layer_type"]
        self.region_graph = self.config["region_graph"]
        self.num_units = self.config["num_units"]
        self.lr = self.config["lr"]
        self.image_size = self.config["image_size"]
        self.channel = self.config["channel"]
        if load_composite:
            # print(f"Finished loading patch level", flush=True)
            return
        self.patch_fn = patchify(self.kernel_size, self.kernel_size, compile=False)
        self.patch_per_img = (self.image_size[0] // self.kernel_size[0]) * (
            self.image_size[1] // self.kernel_size[1]
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)

        log_likelihoods = self.circuit(batch)
        loss = -torch.mean(log_likelihoods)

        if self.circuit_type == "patch":
            self.log("train_loss", (loss * self.patch_per_img).item())
        else:
            self.log("train_loss", loss.item())
        return loss

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

    def on_validation_epoch_end(self):
        val_loss = sum(self.running_loss) / len(self.running_loss)
        self.log("val_loss", val_loss)

    def on_test_epoch_start(self):
        self.test_loss = 0

    def prepare_batch(self, batch):
        if (
            isinstance(batch, list)
            or isinstance(batch, tuple)
            or isinstance(batch, dict)
        ):
            batch, _ = batch

        # this is the test loop
        if self.circuit_type == "patch":
            batch = self.patch_fn(batch)
        BS = batch.shape[0]
        batch = batch.reshape(BS, -1).contiguous()
        return batch

    def test_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        log_likelihoods = self.circuit(batch)
        test_lls = -log_likelihoods.sum().item()
        self.test_loss += test_lls

    def on_test_epoch_end(self):
        total_test_samples = len(self.trainer.datamodule.test_dataloader().dataset)
        average_nll = self.test_loss / total_test_samples
        self.log("test_loss", average_nll)
        self.log("test_bpd", self.get_bpd(average_nll))
        self.logger.save()
        unpatch_fn = unpatchify(
            self.image_size, self.kernel_size, self.kernel_size, self.channel
        )
        if self.layer_type != "tucker" and False:
            query = SamplingQuery(self.circuit)
            num_samples = 5

            sample_list = []
            for _ in range(num_samples):
                num_samples = 1
                if self.circuit_type == "patch":
                    num_samples *= self.patch_per_img
                    print(f"Num patch to sample : {num_samples}")
                samples, mixed = query(num_samples=num_samples)

                samples = samples.cpu()
                img_samples = unpatch_fn(samples)  # (N, C, H, W)
                sample_list.append(img_samples)
            self.logger.log_samples(torch.concat(sample_list))

    def get_bpd(self, average_nll: float):
        img_dim = self.channel * self.image_size[0] * self.image_size[1]
        return average_nll / (img_dim * np.log(2.0))

    def configure_optimizers(self):
        if "weight_decay" in self.config:
            optimizer = optim.Adam(
                self.parameters(), lr=float(self.config["weight_decay"])
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if "scheduler" in self.config and self.config["scheduler"] is not False:
            config_sc = self.config["scheduler"]
            if config_sc["name"] == "cosine_annealing":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=int(config_sc["T_0"]),
                    T_mult=int(config_sc["T_mult"]),
                    eta_min=float(config_sc["eta_min"]),
                )
                return {"optimizer": optimizer, "scheduler": scheduler}
            else:
                raise KeyError(f"Scheduler {config_sc['name']} is not defined")
        return optimizer


def benchmark(config, datamodule, turn_off_logs=False):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    early_stopping_delta = config.get("early_stopping_delta", 0)
    light_mode = BenchPCImage(config)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", min_delta=early_stopping_delta
    )
    exception_callback = StatusExceptionCallback()
    green_callback = GreenCallback()
    logger = new_csv_logger(light_mode.key, exp_dir=config["experiment_path"])

    log_conf = {}
    if turn_off_logs:
        log_conf["enable_progress_bar"] = False
        log_conf["enable_model_summary"] = False

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        logger=logger,
        max_epochs=300,
        callbacks=[early_stop, checkpoint_callback, exception_callback, green_callback],
        num_sanity_val_steps=10,
        log_every_n_steps=400,
        inference_mode=False,
        enable_progress_bar=False,
        precision="bf16-mixed",
        **log_conf,
    )

    try:
        # print("Start Training", flush=True)
        trainer.fit(light_mode, datamodule=datamodule)
        light_mode = load_checkpoint(config, checkpoint_callback.best_model_path)
        test_results = trainer.test(light_mode, datamodule=datamodule)
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
    if config["circuit_type"] == "composite":
        # print(f"Loading cpatch checkpoint {config['patch_ckpt']}", flush=True)
        patch_config = torch.load(config["patch_ckpt"], map_location="cpu")[
            "hyper_parameters"
        ]["config"]
        config["dataset"] = patch_config["dataset"]
        config["early_stopping_delta"] = patch_config.get("early_stopping_delta", 0)
        config["colour_transform"] = patch_config.get("colour_transform", None)
    # print(f"Loading dataset {config['dataset']}", flush=True)

    if config["dataset"] == "mnist":
        datamodule = load_mnist(config)
    elif config["dataset"] == "cifar":
        datamodule = load_cifar(config)
    elif config["dataset"] == "celeba":
        datamodule = load_celebA(config)
    elif config["dataset"] == "imagenet":
        datamodule = load_imagenet(config)
    else:
        print(f"Error: no dataset named {config['dataset']}")
        return
    turn_off_logs = False
    if os.getenv("IN_RAY_TASK") == "1":
        turn_off_logs = True
    # print(f"Dataset {config['dataset']} loaded", flush=True)

    return benchmark(config, datamodule, turn_off_logs=turn_off_logs)


def get_transform(config):
    colour_transform = config.get("colour_transform", None)
    if colour_transform == "ycc_lossless":
        return Rgb2GenT()
    return None


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

    return MNIST(DATASET_ROOT, batch_size, get_transform(config))


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
    return CIFAR(DATASET_ROOT, batch_size, get_transform(config))


def load_celebA(config):
    if "batch_size" in config:
        batch_size = config["batch_size"]
    else:
        batch_size = 64
    config.update(
        {
            "image_size": (64, 64),
            "channel": 3,
        }
    )

    return CelebA(DATASET_ROOT, batch_size, get_transform(config))


def load_imagenet(config):
    if "batch_size" in config:
        batch_size = config["batch_size"]
    else:
        batch_size = 64
    config.update(
        {
            "image_size": (64, 64),
            "channel": 3,
        }
    )

    return ImageNet(DATASET_ROOT, batch_size, get_transform(config))


def load_checkpoint(config, ckpt_path):
    if config.get("use_pic", False):
        ckpt = torch.load(ckpt_path)
        renamed_dict = {}
        for key, value in ckpt["state_dict"].items():
            if "probs" in key and "tensor_parameter" in key:
                continue
            renamed_dict[key.replace("circuit.", "")] = value.clone()

        module = BenchPCImage(config)
        module.circuit.load_state_dict(renamed_dict, strict=False)
        return module
    else:
        BenchPCImage.load_from_checkpoint(ckpt_path)
