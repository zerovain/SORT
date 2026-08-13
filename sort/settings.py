"""
Global settings for SORT.

Provides a central configuration object for package-wide settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """
    Global configuration for SORT.
    
    Attributes
    ----------
    backend : {'auto', 'cupy', 'numpy'}, default='auto'
        Computing backend. 'auto' selects based on availability.
    device : {'auto', 'cuda', 'cpu'}, default='auto'
        Device for computation. 'auto' uses GPU if available.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    seed : int, default=0
        Random seed for reproducibility.
    verbosity : int, default=1
        Verbosity level:
        - 0: Silent (no output)
        - 1: Info (standard progress)
        - 2: Debug (detailed information)
    spatial_key : str, default='spatial'
        Default key in `.obsm` for spatial coordinates.
    loadings_key : str, default='X_sort'
        Default key in `.obsm` for storing loadings.
    signatures_key : str, default='sort_signatures'
        Default key in `.varm` for storing signatures.
    uns_key : str, default='sort'
        Default key in `.uns` for storing metadata.
    
    Examples
    --------
    >>> import sort
    >>> 
    >>> # Use GPU acceleration
    >>> sort.settings.device = 'cuda'
    >>> 
    >>> # Increase verbosity
    >>> sort.settings.verbosity = 2
    >>> 
    >>> # Set random seed
    >>> sort.settings.seed = 42
    >>> 
    >>> # Check current settings
    >>> print(sort.settings)
    """
    
    # Computing backend
    backend: str = 'auto'
    device: str = 'auto'
    n_jobs: int = -1
    
    # Reproducibility
    seed: int = 0
    
    # Verbosity
    verbosity: int = 1
    
    # AnnData keys (customizable for compatibility)
    spatial_key: str = 'spatial'
    loadings_key: str = 'X_sort'
    signatures_key: str = 'sort_signatures'
    uns_key: str = 'sort'
    
    # Display settings
    plot_dpi: int = 100
    plot_format: str = 'png'
    
    def __repr__(self):
        return (
            "SORT Settings\n"
            "="*50 + "\n"
            f"Computing:\n"
            f"  backend      : {self.backend}\n"
            f"  device       : {self.device}\n"
            f"  n_jobs       : {self.n_jobs}\n"
            f"\nReproducibility:\n"
            f"  seed         : {self.seed}\n"
            f"\nOutput:\n"
            f"  verbosity    : {self.verbosity}\n"
            f"\nAnnData keys:\n"
            f"  spatial_key  : {self.spatial_key}\n"
            f"  loadings_key : {self.loadings_key}\n"
            f"  signatures_key: {self.signatures_key}\n"
            f"  uns_key      : {self.uns_key}\n"
            "="*50
        )


# Global settings instance
settings = Settings()