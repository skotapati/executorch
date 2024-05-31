#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSDequantizePerChannelGroup,
    MPSGraph,
    MPSDataType,
    MPSNode
)
from typing import cast, List
from executorch.backends.transforms import get_shape
from executorch.backends.apple.mps.utils.mps_utils import get_input_node, get_scalar_val

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

@register_node_visitor
class OpDequantizePerChannelGroupDefault(NodeVisitor):
    target = "quantized_decomposed.dequantize_per_channel_group.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        print(node)
        print(node.args)
        print("Shapes:")
        print(get_shape(node))
        print(get_shape(node.args[0]))
        print(get_shape(node.args[1]))
        print(get_shape(node.args[2]))
        print(node.meta["val"])
        # omposed.dequantize_per_channel_group.default>: schema = quantized_decomposed::dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tenso
        # Weights placeholders shouldn't have been defined until this point
        if get_input_node(node, 0) in self.tensor_to_id:
            raise RuntimeError(f"Placeholder for {node.target.__name__} already visited")
        print(">>> Defining outout tensor")
        output_id = self.define_tensor(node, mps_graph)
        print(">>> Defining outout tensor")
        input_id = self.define_tensor(get_input_node(node, 0), mps_graph, MPSDataType.mps_data_type_int4)
        print(">>> Defining scales tensor")
        scales_id = self.define_tensor(get_input_node(node, 1), mps_graph)
        print(">>> Defining zero points tensor")
        # there are no zero points in this quantization method (all is zeros)
        # don't pack this data
        # zero_points_id = self.define_tensor(get_input_node(node, 2), mps_graph, MPSDataType.mps_data_type_int4)
        zero_points_id = -1
        quant_min = cast(int, node.args[3])
        quant_max = cast(int, node.args[4])
        dtype = self.torch_dtype_to_mps_dtype(node.args[5])
        group_size = cast(int, node.args[6])
        output_dtype = self.torch_dtype_to_mps_dtype(node.args[7])


        dequant_node = MPSNode(
            mpsnode_union=MPSDequantizePerChannelGroup(
                input1_id=input_id,
                output_id=output_id,
                scales_id=scales_id,
                zero_points_id=zero_points_id,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=dtype,
                group_size=group_size,
                output_dtype=output_dtype,
            )
        )
        mps_graph.mps_nodes.append(dequant_node)
        # print(dequant_node)
        # print(mps_graph)
        # assert(False)
