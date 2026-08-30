from .principal_angles import (
    GeometryGate,
    GeometryGateResult,
    orthonormal_basis,
    principal_angles_deg,
    project_out,
)

__all__ = [
    "GeometryGate",
    "GeometryGateResult",
    "orthonormal_basis",
    "principal_angles_deg",
    "project_out",
]
from .task_conditioned import (
    TaskConditionedGeometry,
    TaskContrast,
    fit_task_conditioned_geometry,
)
from .contrastive import ContrastiveAxis, fit_contrastive_axis
from .residual import fit_residual_stream_axes, last_token_residual_stack
from .consensus import ConsensusSubspace, grassmann_consensus
from .fisher import fisher_factor_from_gradients
from .leace import LeaceEraser, fit_leace
from .metric import (
    LowRankMetric,
    MetricGeometryGate,
    metric_orthonormal_basis,
    metric_principal_angles_deg,
    metric_residualize,
    require_static_geometry,
)
from .pca import PrincipalComponentFit, exact_principal_components
from .token_positions import (
    PromptTokenPositions,
    instruction_index_from_offsets,
    locate_prompt_positions,
)

__all__ = [
    "PromptTokenPositions",
    "PrincipalComponentFit",
    "ConsensusSubspace",
    "ContrastiveAxis",
    "LeaceEraser",
    "LowRankMetric",
    "MetricGeometryGate",
    "TaskConditionedGeometry",
    "TaskContrast",
    "fit_task_conditioned_geometry",
    "exact_principal_components",
    "fisher_factor_from_gradients",
    "fit_leace",
    "fit_contrastive_axis",
    "fit_residual_stream_axes",
    "grassmann_consensus",
    "instruction_index_from_offsets",
    "locate_prompt_positions",
    "last_token_residual_stack",
    "metric_orthonormal_basis",
    "metric_principal_angles_deg",
    "metric_residualize",
    "require_static_geometry",
]
