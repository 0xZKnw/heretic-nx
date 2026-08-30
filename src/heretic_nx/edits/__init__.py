from .cayley import apply_cayley, cayley_matrix
from .projector import ProjectorFactors, projector_factors

__all__ = ["ProjectorFactors", "apply_cayley", "cayley_matrix", "projector_factors"]
from .activation_op import (
    ActivationOperator,
    activation_forward_hook,
    metric_projector_operator,
)
from .affine import AffineActivationOperator, affine_operator_from_leace
from .matrix_opt import LowRankOptimizationResult, fit_low_rank_matrix_operator
from .norm_preserving import norm_preserving_weight_edit
from .residual_stream import (
    ResidualStreamWeightEditor,
    apply_residual_stream_weight_edits,
    build_residual_stream_weight_editors,
    snapshot_residual_stream_weights,
)
from .nx_ir2 import (
    ActivationEditIR,
    NXIR2,
    RiskProbeIR,
    RoutePolicyIR,
    SemanticSiteRef,
    ThinkClosePolicyIR,
    TimePolicyIR,
)
from .nx_ir3 import ArtifactReferenceIR, EditIR3, GateEvidenceIR, NXIR3
from .gguf_q8 import (
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
    inspect_q8_gguf,
)
from .gguf_quant import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
    inspect_quantized_gguf,
)
from .sparse import atomic_unit_scores, select_atomic_units
from .spectral import SignedSpectralEdit, fit_signed_spectral_operator

__all__ = [
    "ActivationEditIR",
    "ActivationOperator",
    "AffineActivationOperator",
    "ArtifactReferenceIR",
    "EditIR3",
    "GateEvidenceIR",
    "GGUFQ8AblationPlan",
    "GGUFQ8TensorEdit",
    "GGUFQuantizedAblationPlan",
    "GGUFQuantizedTensorEdit",
    "LowRankOptimizationResult",
    "NXIR2",
    "NXIR3",
    "RoutePolicyIR",
    "RiskProbeIR",
    "ResidualStreamWeightEditor",
    "SemanticSiteRef",
    "TimePolicyIR",
    "ThinkClosePolicyIR",
    "activation_forward_hook",
    "apply_q8_gguf_ablation",
    "apply_quantized_gguf_ablation",
    "apply_residual_stream_weight_edits",
    "affine_operator_from_leace",
    "atomic_unit_scores",
    "build_residual_stream_weight_editors",
    "metric_projector_operator",
    "norm_preserving_weight_edit",
    "fit_low_rank_matrix_operator",
    "fit_signed_spectral_operator",
    "inspect_q8_gguf",
    "inspect_quantized_gguf",
    "select_atomic_units",
    "snapshot_residual_stream_weights",
    "SignedSpectralEdit",
]
