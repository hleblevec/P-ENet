from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (  # ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    SortGraph,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.infer_data_layouts import InferDataLayouts

import finn.transformation.streamline.absorb as absorb
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarMulPastConvTranspose, MoveScalarLinearPastInvariants
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import InferPixelPaddingDeconv
from finn.transformation.streamline.reorder import MoveTransposePastJoinAdd, MoveTransposePastFork, MakeScaleResizeNHWC
from finn.transformation.streamline.reorder import MoveMulPastFork, MoveLinearPastEltwiseAdd, MoveScalarMulPastConv, MoveScalarMulPastConvTranspose
from finn.transformation.streamline.absorb import AbsorbMulIntoMultiThreshold
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.builder.build_dataflow_steps import verify_step
from finn.builder.build_dataflow_config import VerificationStepType


from finn.builder.build_dataflow_config import DataflowBuildConfig


def custom_step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """

    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(MoveLinearPastEltwiseAdd())
    model = model.transform(MoveMulPastFork())
    model = model.transform(MoveScalarMulPastConv())
    model = model.transform(MoveScalarMulPastConvTranspose())
    model = model.transform(AbsorbMulIntoMultiThreshold())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(InferDataLayouts())
    need_convtranspose = len(model.get_nodes_by_op_type("ConvTranspose")) > 0
    if need_convtranspose:
        model = model.transform(InferPixelPaddingDeconv())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(InferDataLayouts())
    model = model.transform(MakeScaleResizeNHWC())
    for i in range(18):
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(MoveTransposePastJoinAdd())
        model = model.transform(MoveTransposePastFork())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(Streamline())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def custom_step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HLSCustomOp` subclasses that represent HLS
    layers. Which nodes and particular configurations can be converted to HLS
    is limited, see the source code of the `convert_to_hw` module for more."""

    model.set_tensor_datatype(model.graph.input[0].name, DataType["INT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferConvInpGen())      
    model = model.transform(to_hw.InferUpsample())
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model
