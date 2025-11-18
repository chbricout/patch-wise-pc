from copy import copy

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import InputLayer, SumLayer
from cirkit.symbolic.parameters import (
    Parameter,
    ReferenceParameter,
    TensorParameter,
)
from cirkit.utils.scope import Scope


def copy_parameter(graph: Parameter):
    new_param_nodes = []
    copy_map = {}
    in_nodes = {}
    outputs = []
    for n in graph.topological_ordering():
        instance = type(n)
        if instance == TensorParameter:
            new_param = ReferenceParameter(n)
        else:
            new_param = instance(**n.config)
        new_param_nodes.append(new_param)
        copy_map[n] = new_param
        inputs = [copy_map[in_node] for in_node in graph.node_inputs(n)]
        if len(inputs) > 0:
            in_nodes[new_param] = inputs
    outputs = [copy_map[out_node] for out_node in graph.outputs]
    return Parameter(nodes=new_param_nodes, in_nodes=in_nodes, outputs=outputs)


def copy_circuit(graph: Circuit, scope_map: dict[int, int], root_node_outputs=None):
    new_circ_layers = []
    copy_map = {}
    in_nodes = {}
    outputs = []
    copied_params = []
    for layer in graph.topological_ordering():
        instance = type(layer)
        if isinstance(layer, SumLayer):
            if layer.weight in copied_params:
                parameter = copy_parameter(layer.weight)
            else:
                parameter = layer.weight
                copied_params.append(layer.weight)
            new_config = layer.config
            new_layer = SumLayer(**new_config, weight=parameter)
        if isinstance(layer, InputLayer):
            params = list(layer.params.items())
            p_key = params[0][0]
            p_graph = params[0][1]
            new_scope = Scope([scope_map[s] for s in layer.scope])
            config = copy(layer.config)
            del config["scope"]
            if p_graph in copied_params:
                new_p_graph = copy_parameter(p_graph)
            else:
                new_p_graph = p_graph
                copied_params.append(p_graph)
            new_layer = instance(scope=new_scope, **config, **{p_key: new_p_graph})
        else:
            new_config = layer.config
            new_layer = instance(**new_config, **layer.params)
        new_circ_layers.append(new_layer)
        copy_map[layer] = new_layer
        inputs = [copy_map[in_node] for in_node in graph.node_inputs(layer)]
        if len(inputs) > 0:
            in_nodes[new_layer] = inputs
    outputs = [copy_map[out_node] for out_node in graph.outputs]
    if root_node_outputs is not None:
        for n in outputs:
            n.num_output_units = root_node_outputs
    return new_circ_layers, in_nodes, outputs


def share_scope(big_circ: Circuit, share_small: Circuit, scope_size: int):
    layers = copy(big_circ.nodes)
    in_layers = copy(big_circ.nodes_inputs)
    layers_to_replace = []
    for n in big_circ.layerwise_topological_ordering():
        if len(big_circ.layer_scope(n[0])) == scope_size and isinstance(n[0], SumLayer):
            layers_to_replace = n
            break

    entry_points_map = {}
    for l in layers_to_replace:
        for sl in big_circ.subgraph(l).topological_ordering():
            to_remove = [sl]
            while len(to_remove) > 0:
                tr = to_remove.pop()
                if tr in in_layers:
                    to_remove.extend(in_layers[tr])
                    del in_layers[tr]
                if tr in layers:
                    layers.remove(tr)

        scope_map = dict(zip(share_small.scope, big_circ.layer_scope(l)))
        new_subgraph_layers, new_subgraph_inputs, new_output = copy_circuit(
            share_small, scope_map, root_node_outputs=sl.num_output_units
        )
        layers.extend(new_subgraph_layers)
        in_layers.update(new_subgraph_inputs)

        entry_points_map[sl] = new_output

    for old, new in entry_points_map.items():
        for node, inputs in in_layers.items():
            if old in inputs:
                in_layers[node].remove(old)
                in_layers[node].extend(new)

    return Circuit(layers=layers, in_layers=in_layers, outputs=big_circ.outputs)
