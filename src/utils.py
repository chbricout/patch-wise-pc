import os
import random
import shutil
import socket
import sys
import uuid
from datetime import datetime
from typing import Optional, Union
import functools
from src.monarch_hclt import construct_hclt_vtree
import torch
import yaml
from cirkit.templates import data_modalities, utils
from cirkit.templates.region_graph import ChowLiuTree
from cirkit.symbolic.parameters import mixing_weight_factory
from lightning import Callback
from pytorch_lightning.loggers import CometLogger, CSVLogger
from ray import tune
from torch.utils.flop_counter import FlopCounterMode
from torchvision.utils import save_image
import numpy as np
from src.orchestrator import get_config_key

from collections.abc import Callable
from os import PathLike
from pathlib import Path

import graphviz

from cirkit.templates.region_graph.graph import PartitionNode, RegionGraph, RegionNode
from cirkit.templates.region_graph.algorithms.utils import tree2rg


def plot_region_graph(
    region_graph: RegionGraph,
    out_path: str | PathLike[str] | None = None,
    orientation: str = "vertical",
    region_node_shape: str = "box",
    partition_node_shape: str = "point",
    label_font: str = "times italic bold",
    label_size: str = "21pt",
    label_color: str = "white",
    region_label: str | Callable[[RegionNode], str] | None = None,
    region_color: str | Callable[[RegionNode], str] = "#607d8b",
    partition_label: str | Callable[[PartitionNode], str] | None = None,
    partition_color: str | Callable[[PartitionNode], str] = "#ffbd2a",
) -> graphviz.Digraph:
    """Plot the region graph using graphviz.

    Args:
        region_graph: The region graph to plot.
        out_path: The output path where the plot is saved.
            If it is None, the plot is not saved to a file. Defaults to None.
            The Output file format is deduced from the path. Possible formats are:
            {'jp2', 'plain-ext', 'sgi', 'x11', 'pic', 'jpeg', 'imap', 'psd', 'pct',
             'json', 'jpe', 'tif', 'tga', 'gif', 'tk', 'xlib', 'vmlz', 'json0', 'vrml',
             'gd', 'xdot', 'plain', 'cmap', 'canon', 'cgimage', 'fig', 'svg', 'dot_json',
             'bmp', 'png', 'cmapx', 'pdf', 'webp', 'ico', 'xdot_json', 'gtk', 'svgz',
             'xdot1.4', 'cmapx_np', 'dot', 'tiff', 'ps2', 'gd2', 'gv', 'ps', 'jpg',
             'imap_np', 'wbmp', 'vml', 'eps', 'xdot1.2', 'pov', 'pict', 'ismap', 'exr'}.
             See https://graphviz.org/docs/outputs/ for more.
        orientation: Orientation of the graph. "vertical" puts the root
            node at the top, "horizontal" at left. Defaults to "vertical".
        label_font: Font used to render labels. Defaults to "times italic bold".
            See https://graphviz.org/faq/font/ for the available fonts.
        label_size: Size of the font for labels in points. Defaults to "21pt".
        label_color: Color for the labels in the nodes. Defaults to "white".
            See https://graphviz.org/docs/attr-types/color/ for supported color.
        region_label: Either a string or a function.
            If a function is provided, then it must take as input a region node and returns a string
            that will be used as label. If None, it defaults to the string representation of the
            scope of the region node.
        region_color: Either a string or a function.
            If a function is provided, then it must take as input a region node and returns a string
            that will be used as color for the region node. Defaults to "#607d8b".
        partition_label: Either a string or a
            function. If a function is provided, then it must take as input a partition node and
            returns a string that will be used as label. If None, it defaults to an empty string.
        partition_color: Either a string or a function.
            If a function is provided, then it must take as input a partition node and returns a
            string that will be used as color for the partition node. Defaults to "#ffbd2a".

    Raises:
        ValueError: The format is not among the supported ones.
        ValueError: The direction is not among the supported ones.

    Returns:
        graphviz.Digraph: The graphviz object representing the region graph.
    """
    fmt: str
    if out_path is None:
        fmt = "svg"
    else:
        fmt = Path(out_path).suffix.replace(".", "")
        if fmt not in graphviz.FORMATS:
            raise ValueError(f"Supported formats are {graphviz.FORMATS}.")

    if orientation not in ["vertical", "horizontal"]:
        raise ValueError(
            "Supported graph directions are only 'vertical' and 'horizontal'."
        )

    def _default_region_label(rgn: RegionNode) -> str:
        return str(set(rgn.scope))

    def _default_partition_label(_: PartitionNode) -> str:
        return ""

    if region_label is None:
        region_label = _default_region_label
    if partition_label is None:
        partition_label = _default_partition_label

    dot: graphviz.Digraph = graphviz.Digraph(
        format=fmt,
        node_attr={
            "style": "filled",
            "fontcolor": label_color,
            "fontsize": label_size,
            "fontname": label_font,
        },
        engine="dot",
    )
    dot.graph_attr["rankdir"] = "BT" if orientation == "vertical" else "LR"

    for node in region_graph.nodes:
        match node:
            case RegionNode():
                dot.node(
                    str(id(node)),
                    region_label
                    if isinstance(region_label, str)
                    else region_label(node),
                    color=region_color
                    if isinstance(region_color, str)
                    else region_color(node),
                    shape=region_node_shape,
                )
            case PartitionNode():
                dot.node(
                    str(id(node)),
                    partition_label
                    if isinstance(partition_label, str)
                    else partition_label(node),
                    color=(
                        partition_color
                        if isinstance(partition_color, str)
                        else partition_color(node)
                    ),
                    shape=partition_node_shape,
                    width="0.2",
                )
        for node_in in region_graph.node_inputs(node):
            dot.edge(str(id(node_in)), str(id(node)))
    if out_path is not None:
        out_dir: Path = Path(out_path).with_suffix("")

        if fmt == "dot":
            with open(out_dir, "w", encoding="utf8") as f:
                f.write(dot.source)
        else:
            dot.format = fmt
            dot.render(out_dir, cleanup=True)

    return dot


def patch_circuit_factory(
    kernel_size, channel, region_graph, layer_type, num_units, num_classes=1, **kwargs
):
    if region_graph == "chow-liu-tree":
        if kwargs.get("clt_tree", None) is None:
            raise ValueError('Missing key "clt_tree" to create the CLT graph')
        return chow_liu_tree_factory(
            kwargs.pop("clt_tree"), layer_type, num_units, num_classes, **kwargs
        )
    return data_modalities.image_data(
        (channel, *kernel_size),
        region_graph=region_graph,
        input_layer="categorical",
        num_input_units=num_units,
        sum_product_layer=layer_type,
        num_sum_units=num_units,
        num_classes=num_classes,
        **get_parameterization(**kwargs),
    )


def chow_liu_tree_factory(tree, layer_type, num_units, num_classes=1, **kwargs):
    # if kwargs.get("data", None) is None:
    #     raise ValueError('Missing key "data" to create the CLT graph')
    # data = kwargs["data"].reshape((-1, kernel_size[0] * kernel_size[1] * channel))
    # print("Building HCLT Tree structure")
    # tree, _ = construct_hclt_vtree(data, num_bins=64, sigma=0.02, chunk_size=32)
    print("Converting the tree to a Region Graph")
    rg = tree2rg(tree)
    # rg = ChowLiuTree(
    #     data,
    #     input_type="categorical",
    #     as_region_graph=True,
    #     num_categories=256,
    #     num_bins=8,
    #     chunk_size=10_000,
    # )

    # plot_region_graph(rg, "clt-rg.png")
    use_em = kwargs.get("optimizer", None) == "EM"

    sum_weight_factory = utils.parameterization_to_factory(get_sum_param(use_em))
    nary_sum_weight_factory = functools.partial(
        mixing_weight_factory, param_factory=sum_weight_factory
    )
    print("Building the final circuit")
    return rg.build_circuit(
        input_factory=utils.name_to_input_layer_factory(
            "categorical",
            probs_factory=utils.parameterization_to_factory(get_input_param(use_em)),
            num_categories=256,
        ),
        sum_product=layer_type,
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=nary_sum_weight_factory,
        num_input_units=num_units,
        num_sum_units=num_units,
        num_classes=num_classes,
        factorize_multivariate=True,
    )


def get_input_param(use_em: bool):
    if use_em:
        return utils.Parameterization(initialization="uniform", activation="none")
    else:
        return utils.Parameterization(initialization="normal", activation="softmax")


def get_sum_param(use_em: bool):
    if use_em:
        return utils.Parameterization(initialization="uniform", activation="none")
    else:
        return utils.Parameterization(initialization="normal", activation="softmax")


def get_parameterization(use_pic: bool = False, **kwargs):
    if use_pic:
        return {"sum_weight_param": utils.Parameterization(initialization="normal")}
    elif kwargs.get("optimizer", None) == "EM":
        return {
            "sum_weight_param": get_sum_param((True)),
            "input_params": {"probs": get_input_param(True)},
        }
    else:
        return {"sum_weight_param": get_sum_param(False)}


def circuit_factory(circuit_type: str, *, image_size=None, kernel_size=None, **kwargs):
    name = get_config_key(circuit_type=circuit_type, kernel_size=kernel_size, **kwargs)
    if circuit_type == "patch":
        return name, patch_circuit_factory(kernel_size, **kwargs)
    else:
        return name, patch_circuit_factory(image_size, **kwargs)


def dataloader_to_tensor(loader, device=None, num_samples=-1):
    batches = []
    for batch in loader:
        # handle (features, labels) or single-tensor batches
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if device is not None:
            x = x.to(device)
        batches.append(x)
        if len(batches) * x.shape[0] > num_samples and num_samples != -1:
            break
    return torch.cat(batches, dim=0)


def patchify(kernel_size, stride, compile=True, contiguous_output=False):
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    def _patchify(image: torch.Tensor, one_patch=False):
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
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Lh * Lw, C, kh, kw)
        if one_patch:
            n_patch = patches.shape[1]
            patches = patches[torch.arange(B), torch.randint(0, n_patch, (B,))]
        patches = patches.reshape(-1, C, kh, kw)

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
        self.can_log_status = False
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

    def log_kld(self, kld: np.ndarray):
        np.savetxt(os.path.join(self.log_dir, "kld_sum_weight.csv"), kld, delimiter=",")

    def _compress(self):
        output = os.path.join(self.log_dir, self.name)
        shutil.make_archive(output, "zip", self.log_dir)

    def finalize(self, status: str):
        """Called by Lightning on success or interruption."""

        # Only write the status if we have trained the model
        # This avoid writing the status when running the init time validation loop
        if self.can_log_status:
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
