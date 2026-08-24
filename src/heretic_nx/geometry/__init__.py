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
from .consensus import ConsensusSubspace, grassmann_consensus
from .fisher import fisher_factor_from_gradients
from .leace import LeaceEraser, fit_leace
from .metric import (
    LowRankMetric,
    MetricGeometryGate,
    metric_orthonormal_basis,
    metric_principal_angles_deg,
    metric_residualize,
)
from .token_positions import (
    PromptTokenPositions,
    instruction_index_from_offsets,
    locate_prompt_positions,
)

__all__ = [
    "PromptTokenPositions",
    "ConsensusSubspace",
    "LeaceEraser",
    "LowRankMetric",
    "MetricGeometryGate",
    "TaskConditionedGeometry",
    "TaskContrast",
    "fit_task_conditioned_geometry",
    "fisher_factor_from_gradients",
    "fit_leace",
    "grassmann_consensus",
    "instruction_index_from_offsets",
    "locate_prompt_positions",
    "metric_orthonormal_basis",
    "metric_principal_angles_deg",
    "metric_residualize",
]
