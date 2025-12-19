from optuna import Trial
from ray import tune

mnist_sampling = {
    "name": "MNIST_patch_search",
    "circuit_type": "patch",
    "dataset": "mnist",
    "batch-size": 128,
    "layer_type": tune.choice(
        [
            "cp-t",
            "cp",
        ]
    ),
    "region_graph": tune.choice(["quad-graph", "quad-tree-2"]),
    "num_units": tune.lograndint(10, 1024),
    "lr": tune.loguniform(1e-4, 1e-1),
    "kernel_size": tune.choice([(4, 4), (2, 2), (7, 7), (14, 14), (28, 28)]),
}

cifar_sampling = {
    "name": "CIFAR_patch_search",
    "circuit_type": "patch",
    "dataset": "cifar",
    "batch-size": 128,
    "layer_type": tune.choice(
        [
            "cp-t",
            "cp",
        ]
    ),
    "region_graph": tune.choice(["quad-graph", "quad-tree-2"]),
    "num_units": tune.lograndint(10, 1024),
    "lr": tune.loguniform(1e-4, 1e-1),
    "kernel_size": tune.choice([(4, 4), (2, 2), (8, 8), (16, 16), (32, 32)]),
}


def cifar_config_define(trial: Trial):
    layer_type = trial.suggest_categorical("layer_type", ["cp-t", "cp"])
    region_graph = trial.suggest_categorical(
        "region_graph", ["quad-graph", "quad-tree-2"]
    )

    kernel_size = trial.suggest_categorical("kernel_size", [2, 4, 8, 16, 32])
    if kernel_size < 32:
        num_units = trial.suggest_int("num_units", 10, 1024, log=True)
    else:
        num_units = trial.suggest_int("num_units", 10, 512, log=True)

    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)

    return {
        "circuit_type": "patch",
        "dataset": "cifar",
        "batch-size": 128,
        "layer_type": layer_type,
        "region_graph": region_graph,
        "num_units": num_units,
        "lr": lr,
        "kernel_size": [kernel_size, kernel_size],
    }
