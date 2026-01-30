from typing import Any
from copy import copy
import psutil
import torch
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers.inner import TorchSumLayer
from cirkit.backend.torch.layers.input import TorchInputLayer
from cirkit.backend.torch.layers.optimized import TorchCPTLayer
from cirkit.backend.torch.parameters.nodes import (
    TorchParameterInput,
    TorchTensorParameter,
    TorchUnaryParameterOp,
)
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.symbolic.circuit import Circuit
from cirkit.templates import data_modalities
from cirkit.utils.scope import Scope
from cirkit.pipeline import compile as cirkit_compile

from src.utils import patchify, patch_circuit_factory


def get_share_circ(conf):
    conf_copy = copy(conf)
    top_conf = copy(conf)
    n_patch = tuple(
        i // j for i, j in zip(conf_copy["image_size"], conf_copy["kernel_size"])
    )
    top_conf["kernel_size"] = n_patch
    top_conf["channel"] = 1
    conf_copy["num_classes"] = conf_copy["num_units"]
    patch_symb = patch_circuit_factory(**conf_copy)
    patch_circ = cirkit_compile(patch_symb)
    composite_symb = stitch_circuits(
        top_circuit_param=top_conf, patch_circuit_param=conf_copy
    )
    composite_compiled = cirkit_compile(composite_symb)
    share_param_like(
        composite_compiled,
        patch_circ,
        should_init_mean=False,
        should_freeze=False,
    )

    return composite_compiled


def get_composite_circ(top_circuit_config: dict[str, Any], patch_circuit_module):
    # print(f"Memory Status Beginning {psutil.virtual_memory()}", flush=True)
    top_circ_param = copy(top_circuit_config)
    patch_circuit_config = patch_circuit_module.config
    n_patch = tuple(
        i // j
        for i, j in zip(
            patch_circuit_config["image_size"], patch_circuit_config["kernel_size"]
        )
    )
    top_circ_param["kernel_size"] = n_patch
    top_circ_param["channel"] = 1
    patch_circuit_config["num_classes"] = top_circ_param["num_units"]
    # print(f"Memory Status before stitch {psutil.virtual_memory()}", flush=True)
    composite_symb = stitch_circuits(
        top_circuit_param=top_circ_param, patch_circuit_param=patch_circuit_config
    )
    # print(f"Memory Status after stitch {psutil.virtual_memory()}", flush=True)

    composite_compiled = cirkit_compile(composite_symb)
    # print(f"Memory Status after compile {psutil.virtual_memory()}", flush=True)

    share_param_like(
        composite_compiled,
        patch_circuit_module.circuit,
        should_init_mean=top_circuit_config.get("should_init_mean", False),
        should_freeze=top_circuit_config.get("should_freeze", False),
        sum_noise_magnitude=top_circuit_config.get("sum_noise_magnitude", 0),
    )
    # print(f"Memory Status after share {psutil.virtual_memory()}", flush=True)

    assert (
        test_parameter_sharing(
            composite_compiled,
            patch_circuit_module.circuit,
            (patch_circuit_config["channel"], *patch_circuit_config["image_size"]),
            patch_circuit_config["kernel_size"],
        )
        or top_circuit_config.get("sum_noise_magnitude", None) is not None
    ), "Parameter sharing failed, likelihood is not equal"
    print(f"Memory Status after test {psutil.virtual_memory()}", flush=True)

    return composite_compiled


def stitch_circuits(top_circuit_param, patch_circuit_param):
    im_shape = top_circuit_param["image_size"]
    kernel_shape = patch_circuit_param["kernel_size"]

    index_pixel = torch.arange(
        im_shape[0] * im_shape[1] * patch_circuit_param["channel"]
    ).reshape(1, patch_circuit_param["channel"], *im_shape)
    patch_fn = patchify(kernel_shape, kernel_shape)
    scope_order = patch_fn(index_pixel).reshape(
        -1, kernel_shape[0] * kernel_shape[1] * patch_circuit_param["channel"]
    )
    top = patch_circuit_factory(**top_circuit_param)
    new_layers = top._nodes.copy()
    new_inputs = top._in_nodes.copy()
    for new_scope, input_node in zip(
        scope_order, list(top.layerwise_topological_ordering())[0]
    ):
        # Remove input node
        new_layers.remove(input_node)
        # add output of patch (create patch)
        patch = patch_circuit_factory(**patch_circuit_param)
        patch_input = list(patch.layerwise_topological_ordering())[0]
        for idx, inp in enumerate(patch_input):
            inp.scope = Scope([new_scope[list(inp.scope)[0]].item()])
        new_layers.extend(patch._nodes)

        # verify connections
        for node, inputs in top._in_nodes.items():
            if input_node in inputs:
                new_inputs[node].remove(input_node)
                new_inputs[node].extend(patch.outputs)
        new_inputs.update(patch._in_nodes)

    return Circuit(new_layers, new_inputs, top.outputs)


def copy_parameter(graph: TorchParameter, new_shape):
    new_param_nodes = []
    copy_map = {}
    in_nodes = {}
    outputs = []
    for n in graph.topological_ordering():
        instance = type(n)
        config = n.config
        if isinstance(n, TorchTensorParameter):
            del config["shape"]
            new_param = instance(*new_shape, **config)
            new_param._ptensor = torch.nn.Parameter(
                torch.zeros((graph.shape[0], *new_shape))
            )

        elif isinstance(n, TorchUnaryParameterOp):
            config["in_shape"] = new_shape

            new_param = instance(**config)
        new_param_nodes.append(new_param)
        copy_map[n] = new_param
        inputs = [copy_map[in_node] for in_node in graph.node_inputs(n)]
        if len(inputs) > 0:
            in_nodes[new_param] = inputs
    outputs = [copy_map[out_node] for out_node in graph.outputs]
    parameter = TorchParameter(
        modules=new_param_nodes, in_modules=in_nodes, outputs=outputs
    )
    return parameter


class TorchSharedParameter(TorchParameterInput):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        parameter: list[torch.nn.Module],
        num_folds: int,
    ):
        super().__init__()
        self._num_folds = num_folds
        self.in_shape = in_shape
        self.internal_param = parameter

    def internal_forward(self):
        current_input = None
        for param in self.internal_param:
            if current_input is None:
                current_input = param()
            else:
                current_input = param(current_input)
        return current_input

    def forward(self):
        param = self.internal_forward()
        share_fold, *inner_units = param.shape
        expanded = param.expand(
            self.num_folds // share_fold, share_fold, *inner_units
        ).reshape(self.num_folds, *inner_units)

        return expanded

    @property
    def shape(self):
        return self.in_shape


def share_param_like(
    base_circ: TorchCircuit,
    share_struct: TorchCircuit,
    should_init_mean=False,
    should_freeze=False,
    sum_noise_magnitude=0,
):
    for idx, layer in enumerate(share_struct.layers):
        if isinstance(layer, TorchInputLayer):
            folds = base_circ.layers[idx].probs.num_folds
            shared_param = TorchSharedParameter(
                base_circ.layers[idx].probs.shape,
                parameter=layer.probs.nodes,
                num_folds=folds,
            )
            base_circ.layers[idx].probs = shared_param
        elif isinstance(layer, TorchCPTLayer) or isinstance(layer, TorchSumLayer):
            internal_param = layer.weight.nodes
            has_new_nodes = False
            if layer.num_output_units != base_circ.layers[idx].num_output_units:
                # TODO insert noise when copying the weight
                new_parameter = copy_parameter(
                    layer.weight, base_circ.layers[idx].weight.nodes[0].shape
                )
                _, o, i = internal_param[0]._ptensor.data.shape
                _, goal_o, goal_i = new_parameter.nodes[0]._ptensor.data.shape
                num_input = goal_i // i
                num_output = goal_o // o
                new_parameter.nodes[0]._ptensor.data = (
                    internal_param[0]
                    ._ptensor.data.clone()
                    .repeat((1, num_output, num_input))
                )
                if sum_noise_magnitude != 0:
                    # Inject noise in the sum node
                    old_weight = new_parameter.nodes[0]._ptensor.clone()
                    old_weight_softmaxed = new_parameter().detach().cpu()
                    print(
                        "Weight magnitude:",
                        old_weight.min(),
                        old_weight.max(),
                    )
                    # Rescale the uniform noise to be between -1 and 1 and then scale it with the choosen noise magnitude
                    new_parameter.nodes[0]._ptensor.data += (
                        (torch.rand_like(new_parameter.nodes[0]._ptensor) * 2) - 1
                    ) * sum_noise_magnitude
                    new_weights = new_parameter.nodes[0]._ptensor.clone()
                    new_weights_softmaxed = new_parameter().detach().cpu()
                    print(
                        "Weight magnitude after noise:",
                        new_weights.min(),
                        new_weights.max(),
                        new_weights_softmaxed.shape,
                    )
                    from scipy.stats import entropy

                    print(
                        "KL Divergence (scipy entropy)",
                        entropy(old_weight_softmaxed, new_weights_softmaxed, axis=1),
                        (old_weight_softmaxed - new_weights_softmaxed).abs().sum(),
                    )

                has_new_nodes = True

                internal_param = new_parameter.nodes
            folds = base_circ.layers[idx].weight.num_folds
            shared_param = TorchSharedParameter(
                base_circ.layers[idx].weight.shape,
                parameter=internal_param,
                num_folds=folds,
            )
            if should_freeze and not has_new_nodes:
                freeze_parameter(shared_param)
            base_circ.layers[idx].weight = shared_param
    if should_init_mean:
        init_mean(base_circ, len(share_struct.layers))


def init_mean(circ, start_idx):
    for layer in circ.layers[start_idx:]:
        print(layer)
        if isinstance(layer, TorchCPTLayer) or isinstance(layer, TorchSumLayer):
            param = layer.weight
            tensor = param.nodes[0]._ptensor
            inputs = tensor.shape[-1]

            param.nodes[0]._ptensor.data = torch.full(
                tensor.shape, torch.exp(torch.tensor(1 / inputs))
            )


def freeze_parameter(param: TorchSharedParameter):
    for p in param.parameters():
        p.requires_grad = False


def test_parameter_sharing(
    composite_circ,
    patch_circ,
    image_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
):
    print(f"test patches", flush=True)
    comp_param = sum(p.numel() for p in composite_circ.parameters() if p.requires_grad)
    patch_param = sum(p.numel() for p in patch_circ.parameters() if p.requires_grad)
    print(
        f"Composite circ params: {comp_param}, patch circ params: {patch_param}",
        flush=True,
    )
    with torch.no_grad():
        random_data = torch.randint(256, (1, *image_size), dtype=torch.int32).cuda()

        composite_circ = composite_circ.cuda()
        comp_ll = composite_circ(
            random_data.reshape(-1, image_size[0] * image_size[1] * image_size[2])
        ).item()
        composite_circ = composite_circ.cpu()

        torch.cuda.empty_cache()

        patch_fn = patchify(kernel_size, kernel_size)
        patched = patch_fn(random_data)
        patch_circ = patch_circ.cuda()
        patch_ll = (
            patch_circ(
                patched.reshape(-1, image_size[0] * kernel_size[0] * kernel_size[1])
            )
            .sum()
            .item()
        )
        patch_circ = patch_circ.cpu()

        del patched
        del patch_circ
        del random_data
        torch.cuda.empty_cache()

    print(f"Patch log likelihood: {patch_ll}")
    print(f"Composite log likelihood: {comp_ll}")

    return abs(patch_ll - comp_ll) < 0.01
