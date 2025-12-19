import os
import random

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from src.benchmark_logic import BenchPCImage, load_cifar, load_mnist
from src.utils import GreenCallbackRay


def tune_grid(name: str, config_fn, num_samples=100):
    scheduler = ASHAScheduler(
        max_t=300,
        grace_period=5,
        reduction_factor=2,
        metric="val_loss",
        mode="min",
    )
    algo = OptunaSearch(
        space=config_fn,
        metric="val_loss",
        mode="min",
    )
    train_fn_with_resources = tune.with_resources(
        tune_dataset, resources={"CPU": 1, "GPU": 1}
    )
    storage_path = os.path.abspath("./hp_search")
    tuner = tune.Tuner(
        train_fn_with_resources,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=algo,
        ),
        run_config=tune.RunConfig(
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1, checkpoint_frequency=0
            ),
            name=name,
            storage_path=f"file://{storage_path}",
        ),
    )
    analysis = tuner.fit()
    df = analysis.get_dataframe(filter_metric="val_bpd", filter_mode="max")

    # Save to CSV
    df.to_csv(f"./hp_search/{name}/results_summary.csv", index=False)


def tune_pl(config, train_dataloader, val_dataloader, test_dataloader):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    light_mode = BenchPCImage(config, len(test_dataloader.dataset))
    green_callback = GreenCallbackRay()
    tunereport_callback = TuneReportCheckpointCallback(
        metrics=["val_loss", "val_bpd", "number_parameters"],
        filename="checkpoint.ckpt",
    )
    early_stop = EarlyStopping(monitor="val_loss", mode="min")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        callbacks=[
            green_callback,
            tunereport_callback,
            early_stop,
        ],
        inference_mode=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(
        light_mode,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def tune_dataset(config):
    if config["dataset"] == "mnist":
        dataloaders = load_mnist(config)
    elif config["dataset"] == "cifar":
        dataloaders = load_cifar(config)
    else:
        print(f"Error: no dataset named {config['dataset']}")
        return

    return tune_pl(config, **dataloaders)
