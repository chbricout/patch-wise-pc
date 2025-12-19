from typing import Any

field_requirement = {
    "circuit_type": True,
    "layer_type": True,
    "region_graph": True,
    "num_units": True,
    "lr": True,
    "dataset": True,
    "kernel_size": True,
    "batch_size": True,
    "colour_transform": False,
    "early_stopping_delta": False,
    "use_pic": False,
}


def check_hparam(config: dict[str, Any]):
    for param_key, required in field_requirement.items():
        if param_key not in config and required:
            raise KeyError(f"Config miss the required field: {param_key}")
