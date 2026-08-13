"""Validated configuration for SORT's high-level entry point."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class SORTConfig:
    """Parameters forwarded unchanged to :func:`sort.decompose`.

    Defaults capture the settings shared by most biological applications in
    the manuscript. Dataset-specific component counts, gene selection and
    graph construction remain explicit because changing them changes the
    analysis. Exact manuscript exceptions are retained in the analysis
    archive.
    """

    n_components: int = 25
    layer: Optional[str] = None
    use_highly_variable: bool = True
    alpha: float = 0.3
    beta: float = 0.5
    lambda_l1_W: float = 0.3
    lambda_l1_Q: float = 300.0
    l1_weight_strategy: Literal["fixed", "adaptive", "none"] = "adaptive"
    lambda_neg: float = 1.0
    use_tv: bool = True
    tv_epsilon: float = 1e-2
    tv_update_freq: int = 5
    tv_stage: Literal["stage2", "both"] = "stage2"
    stage1_epochs: int = 50
    stage2_epochs: int = 100
    ortho_mode: str = "huber"
    huber_delta: float = 0.5
    smooth_l1_delta: float = 0.1
    adam_steps: int = 40
    grad_clip_norm: float = 1.0
    device: Literal["auto", "cuda", "cpu", "numpy"] = "auto"
    random_state: Optional[int] = 42
    auto_init: bool = True
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    verbose: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least 2")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be nonnegative")
        if self.lambda_l1_W < 0 or self.lambda_l1_Q < 0 or self.lambda_neg < 0:
            raise ValueError("regularization strengths must be nonnegative")
        if self.tv_epsilon <= 0 or self.tv_update_freq < 1:
            raise ValueError("TV epsilon and update frequency must be positive")
        if self.stage1_epochs < 0 or self.stage2_epochs < 0:
            raise ValueError("epoch counts must be nonnegative")
        if self.adam_steps < 1 or self.grad_clip_norm <= 0:
            raise ValueError("Adam steps and gradient clipping must be positive")
        if self.l1_weight_strategy not in {"fixed", "adaptive", "none"}:
            raise ValueError("unsupported l1_weight_strategy")
        if self.tv_stage not in {"stage2", "both"}:
            raise ValueError("tv_stage must be 'stage2' or 'both'")
        if self.device not in {"auto", "cuda", "cpu", "numpy"}:
            raise ValueError("device must be 'auto', 'cuda', 'cpu', or 'numpy'")

    def to_decompose_kwargs(self) -> Dict[str, Any]:
        """Return a copy suitable for direct forwarding to ``decompose``."""

        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialization-ready copy."""

        return asdict(self)
