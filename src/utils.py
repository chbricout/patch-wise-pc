import os
import shutil
import socket
import sys
import uuid
from datetime import datetime
from typing import Optional, Union

import torch
import yaml
from cirkit.templates import data_modalities, utils
from lightning import Callback
from pytorch_lightning.loggers import CometLogger, CSVLogger
from ray import tune
from torch.utils.flop_counter import FlopCounterMode
from torchvision.utils import save_image

from src.orchestrator import get_config_key


def patch_circuit_factory(
    kernel_size, channel, region_graph, layer_type, num_units, num_classes=1, **kwargs
):
    return data_modalities.image_data(
        (channel, *kernel_size),
        region_graph=region_graph,
        input_layer="categorical",
        num_input_units=num_units,
        sum_product_layer=layer_type,
        num_sum_units=num_units,
        num_classes=num_classes,
        sum_weight_param=get_parameterization(**kwargs),
    )


def get_parameterization(use_pic: bool = False, **kwargs):
    if use_pic:
        return utils.Parameterization(initialization="normal")
    else:
        return utils.Parameterization(activation="softmax", initialization="normal")


def circuit_factory(circuit_type: str, *, image_size=None, kernel_size=None, **kwargs):
    name = get_config_key(circuit_type=circuit_type, kernel_size=kernel_size, **kwargs)
    if circuit_type == "patch":
        return name, patch_circuit_factory(kernel_size, **kwargs)
    else:
        return name, patch_circuit_factory(image_size, **kwargs)


def patchify(kernel_size, stride, compile=True, contiguous_output=False):
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    def _patchify(image: torch.Tensor):
        # Accept (C,H,W) or (B,C,H,W)

        # Ensure contiguous NCHW for predictable strides
        x = image.contiguous()  # (B,C,H,W)
        B, C, H, W = x.shape

        # Number of patches along H/W
        Lh = (H - kh) // sh + 1
        Lw = (W - kw) // sw + 1

        # Create a zero-copy view: (B, C, Lh, Lw, kh, kw)
        sN, sC, sH, sW = x.stride()
        patches = x.as_strided(
            size=(B, C, Lh, Lw, kh, kw),
            stride=(sN, sC, sH * sh, sW * sw, sH, sW),
        )
        # Reorder to (B, P, C, kh, kw) where P = Lh*Lw
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * Lh * Lw, C, kh, kw)

        if contiguous_output:
            patches = (
                patches.contiguous()
            )  # materialize if the next ops need contiguous

        return patches

    if compile:
        _patchify = torch.compile(_patchify, fullgraph=True, dynamic=False)
    return _patchify


def unpatchify(
    image_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    channel: int,
):
    H, W = image_size
    kh, kw = kernel_size
    sh, sw = stride

    def _unpatchify(patches: torch.Tensor):
        """
        patches: (N_patches_total, C*kh*kw) or (N_patches_total, C, kh, kw)
        image_size: (H, W)
        kernel_size: (kh, kw)
        stride: (sh, sw)
        channel: number of channels C
        Returns: (N_images, C, H, W) tensor (float)
        """
        if patches.dim() == 2:
            # flatten -> (N, C, kh, kw)
            patches = patches.view(-1, channel, kh, kw)

        Bpatch = patches.shape[0]

        Lh = (H - kh) // sh + 1
        Lw = (W - kw) // sw + 1
        patches_per_image = Lh * Lw
        assert Bpatch % patches_per_image == 0, (
            "Total patches not divisible by patches_per_image"
        )

        n_images = Bpatch // patches_per_image
        patches = patches.view(n_images, patches_per_image, channel, kh, kw)

        images = torch.zeros(
            (n_images, channel, H, W), dtype=patches.dtype, device=patches.device
        )

        idx = 0
        for ih in range(Lh):
            for iw in range(Lw):
                patch_idx = ih * Lw + iw
                h0 = ih * sh
                w0 = iw * sw
                images[:, :, h0 : h0 + kh, w0 : w0 + kw] = patches[:, patch_idx]
        return images

    return _unpatchify


def new_experiment(key: str):
    experiment_id = uuid.uuid4().hex[:40]
    comet_logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        project="benchmark-mnist",
        workspace="probabilistic-conv",
        experiment_key=experiment_id,
        online=True,
        # offline_directory="./comet_offline"
    )
    comet_logger.experiment.set_name(key)
    return comet_logger


def new_csv_logger(key: str, exp_dir=None):
    if exp_dir is not None:
        save_dir = os.path.join(exp_dir, "runs")
    else:
        save_dir = "./runs"
    csv_logger = LocalLogger(
        save_dir=save_dir,
        name=key,
        # should_compress=True
    )
    return csv_logger


def get_flops(model, inp, with_backward=False):
    istrain = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops


def write_summary_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}

    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)

    return {
        "available": True,
        "device_index": gpu_id,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "multi_processor_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
    }


class LocalLogger(CSVLogger):
    def __init__(
        self,
        save_dir: str,
        name: str,
        compress: bool = False,
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self.summary_path = os.path.join(self.log_dir, "summary.yaml")
        self.start_time = datetime.now()
        self.should_compress = compress
        # Initial record of experiment metadata
        self.summary = {
            "hostname": socket.gethostname(),
            "python_version": sys.version.replace("\n", " "),
            "command": " ".join(sys.argv),
            "start_datetime": self.start_time.isoformat(),
            "gpu": get_gpu_info(),
            "status": "RUNNING",
        }

        write_summary_yaml(self.summary_path, self.summary)

    @property
    def sample_dir(self):
        save_sample_dir = os.path.join(self.log_dir, "samples")

        os.makedirs(save_sample_dir, exist_ok=True)
        return save_sample_dir

    def log_samples(self, samples):
        img_samples = samples / 255.0 if samples.max() > 1 else samples
        for idx, img in enumerate(img_samples):
            file_path = os.path.join(self.sample_dir, f"sample_{idx}.png")
            save_image(img, file_path)

    def _compress(self):
        output = os.path.join(self.log_dir, self.name)
        shutil.make_archive(output, "zip", self.log_dir)

    def finalize(self, status: str):
        """Called by Lightning on success or interruption."""
        self.summary["end_datetime"] = datetime.now().isoformat()

        if status == "success":
            self.summary["status"] = "SUCCESS"
        else:
            # lightning reports ANY interruption as "failed"
            self.summary["status"] = "FAILED"

        write_summary_yaml(self.summary_path, self.summary)

        if self.should_compress:
            self._compress()

        super().finalize(status)


class StatusExceptionCallback(Callback):
    """Adds exception details to summary.yaml if the run fails."""

    def on_exception(self, trainer, pl_module, exception):
        summary_path = trainer.logger.summary_path

        with open(summary_path, "r") as f:
            data = yaml.safe_load(f)

        data["end_datetime"] = datetime.now().isoformat()
        data["status"] = "FAILED"
        data["exception"] = str(exception)

        write_summary_yaml(summary_path, data)


class StatusCallback:
    pass


class GreenCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.memory.reset_peak_memory_stats()

    def on_train_end(self, trainer, pl_module):
        pl_module.logger.experiment.log_metrics(
            {
                "max_train_memory": torch.cuda.memory.max_memory_allocated(
                    pl_module.device
                )
                / 1e9
            }
        )

    def on_test_epoch_start(self, trainer, pl_module):
        torch.cuda.memory.reset_peak_memory_stats()
        self.test_batch = None

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if self.test_batch is None:
            self.test_batch = pl_module.prepare_batch(batch).cpu()

    def on_test_epoch_end(self, trainer, pl_module):
        with torch.enable_grad():
            device_batch = self.test_batch.to(pl_module.device)
            flops_test = get_flops(pl_module.circuit, device_batch)
            flops_train = get_flops(
                pl_module.circuit,
                device_batch,
                with_backward=True,
            )
            del device_batch
            torch.cuda.empty_cache()
        pl_module.logger.experiment.log_metrics(
            {
                "flops_test": flops_test / 1e9,
                "flops_train": flops_train / 1e9,
                "number_parameters": sum(
                    p.numel() for p in pl_module.circuit.parameters() if p.requires_grad
                ),
                "max_test_memory": torch.cuda.memory.max_memory_allocated(
                    pl_module.device
                )
                / 1e9,
            }
        )


class GreenCallbackRay(Callback):
    def on_train_end(self, trainer, pl_module):
        tune.report(
            {
                "number_parameters": sum(
                    p.numel() for p in pl_module.circuit.parameters() if p.requires_grad
                ),
                "val_loss": sum(pl_module.running_loss) / len(pl_module.running_loss),
            },
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = sum(pl_module.running_loss) / len(pl_module.running_loss)
        pl_module.log("val_bpd", pl_module.get_bpd(val_loss))
