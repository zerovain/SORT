"""
Core optimization and utility functions for SORT.
"""

from .optimization import (
    update_W_multiplicative,
    update_Q_adam,
    compute_positive_negative_parts,
)

from .procrustes import (
    update_q0_closed_form,
    update_Qs_procrustes,
    initialize_Q_from_W,
)

from .regularization import (
    compute_l1_weights_fixed,
    compute_l1_weights_adaptive,
    analyze_scale_variation,
)

from .memory_utils import (
    select_backend_strategy,
    check_gpu_memory,
)

from .sparse_utils import (
    convert_to_backend,
)

__all__ = [
    # Optimization
    'update_W_multiplicative',
    'update_Q_adam',
    'compute_positive_negative_parts',
    
    # Procrustes
    'update_q0_closed_form',
    'update_Qs_procrustes',
    'initialize_Q_from_W',
    
    # Regularization
    'compute_l1_weights_fixed',
    'compute_l1_weights_adaptive',
    'analyze_scale_variation',
    
    # Memory
    'select_backend_strategy',
    'check_gpu_memory',
    
    # Sparse
    'convert_to_backend',
]