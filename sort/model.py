"""Main SORT model for spatial transcriptomics decomposition."""  

import numpy as np  
from scipy.sparse import issparse, spmatrix, csr_matrix  
from anndata import AnnData  
from typing import Optional, Literal, Union  

from .core.optimization import (  
    update_W_multiplicative,  
    update_W_hybrid_manual,  
    update_W_hybrid_manual_tv,  
    update_Q_adam,  
    RiemannianAdam,
)  
from .core.procrustes import (  
    update_q0_closed_form,  
    update_Qs_procrustes,  
    initialize_Q_from_W,  
)  
from .core.regularization import (  
    compute_l1_weights_fixed,  
    compute_l1_weights_adaptive,  
    analyze_scale_variation,  
)  
from .core.memory_utils import get_data_placement_strategy  
from .core.sparse_utils import convert_to_backend, GPUDataManager  
from .core.tv_utils import prepare_tv_data_structures, validate_tv_weights  
from .core.tv_utils_optimized import TVWeightManagerMatrixOpt  
from .initialization import initialize_W_symnmf  
from .utils import subsample_indices, compute_relative_error, log_progress  

try:  
    import cupy as cp  
    CUPY_AVAILABLE = True  
except ImportError:  
    CUPY_AVAILABLE = False  

try:  
    import torch  
    TORCH_AVAILABLE = True  
except ImportError:  
    TORCH_AVAILABLE = False  


# ============================================================================  
# Main SORT Model  
# ============================================================================  

class SpatialSemiNMF:  
    """Spatial Semi-NMF with background-signal decomposition."""  

    def __init__(  
        self,  
        n_components: int,  
        # ── core ───────────────────────────────────────────────────────────
        alpha: float = 0.1,  
        beta: float = 0.05,  # Deprecated; retained for older callers.
        # ── regularisation ──────────────────────────────────────────────────  
        lambda_l1_W: float = 0.01,  
        lambda_l1_Q: float = 0.0,           # off by default  
        l1_weight_strategy: Literal['fixed', 'adaptive', 'none'] = 'adaptive',  
        lambda_neg: float = 0.01,  
        # ── spatial ─────────────────────────────────────────────────────────  
        use_tv: bool = False,  
        tv_epsilon: float = 1e-6,  
        tv_update_freq: int = 5,  
        tv_stage: Literal['stage2', 'both'] = 'stage2',  
        # ── advanced optimisation (rarely need changing) ─────────────────────  
        ortho_mode: str = 'huber',  
        huber_delta: float = 0.5,  
        smooth_l1_delta: float = 0.1,  
        adam_steps: int = 10,  
        grad_clip_norm: float = 50.0,  
        # ── runtime ─────────────────────────────────────────────────────────  
        device: Literal['auto', 'cuda', 'cpu', 'numpy'] = 'auto',  
        verbose: bool = True,  
    ):  
        self.r                  = n_components  
        self.alpha              = alpha  
        self.beta               = beta  # Compatibility metadata only.
        self.lambda_l1_W_base   = lambda_l1_W  
        self.lambda_l1_Q_base   = lambda_l1_Q  
        self.l1_weight_strategy = l1_weight_strategy  
        self.lambda_neg         = lambda_neg  
        self.use_tv             = use_tv  
        self.tv_epsilon         = tv_epsilon  
        self.tv_update_freq     = tv_update_freq  
        self.tv_stage           = tv_stage  
        self.ortho_mode         = ortho_mode  
        self.huber_delta        = huber_delta  
        self.smooth_l1_delta    = smooth_l1_delta  
        self.adam_steps         = adam_steps  
        self.grad_clip_norm     = grad_clip_norm  
        self.device             = device  
        self.verbose            = verbose  

        # runtime state  
        self.W                   = None  
        self.Q                   = None  
        self.lambda_l1_W         = None  
        self.lambda_l1_Q         = None  
        self.tv_manager          = None  
        self.data_manager        = None  
        self.strategy            = None  
        self._backend_setup_done = False  
        self.riem_optimizer      = None

    # ------------------------------------------------------------------  
    # Public API  
    # ------------------------------------------------------------------  

    def fit(  
        self,  
        X: Union[np.ndarray, spmatrix],  
        L: spmatrix,  
        W_init: Optional[np.ndarray] = None,  
        Q_init: Optional[np.ndarray] = None,  
        stage1_epochs: int = 30,  
        stage2_epochs: int = 100,  
        subsample_size: int = 50000,  
        auto_init: bool = True,  
        init_kwargs: Optional[dict] = None,  
    ) -> 'SpatialSemiNMF':  
        """Fit the model."""  
        n, p = X.shape  

        if self.verbose:  
            print("\n" + "=" * 70)  
            print("SORT: Spatial Semi-NMF Decomposition")  
            print("=" * 70)  
            print(f"Data: {n:,} observations × {p:,} variables")  
            print(f"Components: {self.r} (1 background + {self.r - 1} signals)")  
            print(f"Spatial: {'TV (matrix-optimized)' if self.use_tv else 'Laplacian'}")  
            if self.use_tv:  
                print(f"  TV update: every {self.tv_update_freq} epochs")  

        self._initialize_parameters(X, L, W_init, Q_init, auto_init, init_kwargs, n, p)  
        self._compute_l1_weights()  

        self._setup_backend(X, L)  
        self._optimize_stage1(X, stage1_epochs, subsample_size)  
        self._optimize_stage2(X, stage2_epochs, subsample_size)  

        if self.verbose:  
            print("\n" + "=" * 70)  
            print("✓ SORT Complete")  
            print("=" * 70)  
            self._print_final_statistics(X)  

        self._finalize()  
        return self  

    # ------------------------------------------------------------------  
    # Initialisation helpers  
    # ------------------------------------------------------------------  

    def _initialize_parameters(self, X, L, W_init, Q_init, auto_init, init_kwargs, n, p):  
        """Initialise W and Q."""  
        if W_init is None:  
            if not auto_init:  
                raise ValueError("W_init required when auto_init=False")  

            if init_kwargs is None:  
                init_kwargs = {}  
            if self.verbose:  
                print("\n" + "=" * 70)  
                print("Initialization")  
                print("=" * 70)  

            kw = init_kwargs.copy()  
            kw.setdefault('backend', 'cupy' if (self.device in ['auto', 'cuda'] and CUPY_AVAILABLE) else 'numpy')  
            kw.setdefault('verbose', self.verbose)  
            W_init, Q_init = initialize_W_symnmf(X, L, self.r, **kw)  

        self.W = W_init.copy()  

        if Q_init is None:  
            if self.verbose:  
                print("Computing Q_init from W_init...")  
            self.Q = initialize_Q_from_W(self.W, X, method='svd')  
        else:  
            self.Q = Q_init.copy()  

        if self.W.shape != (n, self.r):  
            raise ValueError(f"W shape mismatch: {self.W.shape} vs ({n}, {self.r})")  
        if self.Q.shape != (p, self.r):  
            raise ValueError(f"Q shape mismatch: {self.Q.shape} vs ({p}, {self.r})")  

    def _compute_l1_weights(self):  
        """Compute per-component L1 weights."""  
        if self.l1_weight_strategy == 'none':  
            self.lambda_l1_W = None  
            self.lambda_l1_Q = None  
            if self.verbose:  
                print("\nL1 regularization: DISABLED")  
            return  

        if self.verbose:  
            print("\n" + "=" * 70)  
            print("L1 Regularization")  
            print("=" * 70)  
            analyze_scale_variation(self.W, self.Q, verbose=True)  

        if self.l1_weight_strategy == 'fixed':  
            self.lambda_l1_W = compute_l1_weights_fixed(self.r, self.lambda_l1_W_base)  
            self.lambda_l1_Q = compute_l1_weights_fixed(self.r - 1, self.lambda_l1_Q_base)  
        else:  
            self.lambda_l1_W = compute_l1_weights_adaptive(  
                self.W, self.lambda_l1_W_base, method='l2_norm'  
            )  
            Q_s    = self.Q[:, 1:]  
            ratio  = np.linalg.norm(Q_s, axis=0)  
            ratio  = ratio.max() / (ratio.min() + 1e-10)  
            if ratio > 1.5:  
                self.lambda_l1_Q = compute_l1_weights_adaptive(  
                    Q_s, self.lambda_l1_Q_base, method='l2_norm'  
                )  
            else:  
                self.lambda_l1_Q = compute_l1_weights_fixed(self.r - 1, self.lambda_l1_Q_base)  

        if self.verbose and self.lambda_l1_W is not None:  
            print(f"λ_W: [{self.lambda_l1_W.min():.4f}, {self.lambda_l1_W.max():.4f}]")  
            print(f"λ_Q: [{self.lambda_l1_Q.min():.4f}, {self.lambda_l1_Q.max():.4f}]")  

    def _setup_backend(self, X, L):  
        """Setup compute backend (GPU/CPU) and PyTorch tensors for Q."""  
        if self._backend_setup_done:  
            return  

        if self.verbose:  
            print("\n" + "=" * 70)  
            print("Data Placement Strategy")  
            print("=" * 70)  

        backend      = 'gpu' if self.device in ['auto', 'cuda'] else 'cpu'  
        strategy_info = get_data_placement_strategy(X, self.r, backend=backend, verbose=self.verbose)  
        self.strategy = strategy_info  
        X_loc, W_loc  = strategy_info['X_location'], strategy_info['W_location']  

        # X / data manager  
        if X_loc == 'gpu' and CUPY_AVAILABLE:  
            self.data_manager = GPUDataManager(X, backend='gpu', verbose=self.verbose)  
            self.X_backend    = self.data_manager.get_X()  
        elif X_loc == 'cpu' and W_loc == 'gpu' and CUPY_AVAILABLE:  
            self.data_manager = GPUDataManager(X, backend='gpu', verbose=self.verbose)  
            self.X_backend    = X  
        else:  
            self.data_manager = None  
            self.X_backend    = X  

        # W / L / λ backends  
        if W_loc == 'gpu' and CUPY_AVAILABLE:  
            self.L_backend             = convert_to_backend(L, 'cupy', 'csr')  
            self.W_backend             = cp.asarray(self.W, dtype=cp.float32)  
            self.lambda_l1_W_backend   = cp.asarray(self.lambda_l1_W) if self.lambda_l1_W is not None else None  
            self.lambda_l1_Q_backend   = cp.asarray(self.lambda_l1_Q) if self.lambda_l1_Q is not None else None  
        else:  
            self.L_backend           = L  
            self.W_backend           = self.W  
            self.lambda_l1_W_backend = self.lambda_l1_W  
            self.lambda_l1_Q_backend = self.lambda_l1_Q  

        # TV manager  
        if self.use_tv:  
            if self.verbose:  
                print("\nTV Regularization:")  
            backend_str    = 'cupy' if W_loc == 'gpu' else 'numpy'  
            edges, w_static = prepare_tv_data_structures(L, W_loc, verbose=self.verbose)  
            self.tv_manager = TVWeightManagerMatrixOpt(  
                edges=edges, w_static=w_static, epsilon=self.tv_epsilon,  
                backend=backend_str, n_nodes=X.shape[0], verbose=self.verbose,  
            )  

        # PyTorch for Q  
        if not TORCH_AVAILABLE:  
            raise ImportError(  
                "PyTorch is required for Q optimization.\n"  
                "Install it with:  pip install torch"  
            )  
        if self.device == 'cuda' and not torch.cuda.is_available():  
            raise RuntimeError(  
                "device='cuda' requested but no CUDA GPU found. "  
                "Use device='auto' or device='cpu'."  
            )  

        _torch_device = torch.device(  
            'cuda' if (self.device in ['auto', 'cuda'] and torch.cuda.is_available()) else 'cpu'  
        )  
        self.Q_torch      = torch.tensor(self.Q, dtype=torch.float32,
                                        device=_torch_device)
        self.riem_optimizer = None
        if self.verbose:  
            print(f"Q optimization: PyTorch on {_torch_device}")  

        self._backend_setup_done = True  

    # ------------------------------------------------------------------  
    # Optimisation stages  
    # ------------------------------------------------------------------  

    def _optimize_stage1(self, X, epochs, subsample_size):  
        """Stage 1: Hard orthogonality via Procrustes."""  
        if self.verbose:  
            print("\n" + "=" * 70)  
            print("Stage 1: Hard Orthogonality")  
            print("=" * 70)  

        use_tv         = self.use_tv and (self.tv_stage == 'both')  
        use_subsample  = X.shape[0] > 100000  

        for epoch in range(epochs):  
            if use_tv and epoch > 0 and epoch % self.tv_update_freq == 0:  
                self.tv_manager.update(self.W_backend)  

            self._update_W(use_tv_now=use_tv)  

            if epoch % 5 == 0:  
                self._update_Q_procrustes(X, use_subsample, subsample_size)  

            if self.verbose and (epoch % 10 == 0 or epoch == epochs - 1):  
                self._log_metrics(X, epoch, stage=1)  

    def _optimize_stage2(self, X, epochs, subsample_size):
        """Stage 2: Soft orthogonality via Adam — full data, no subsampling."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("Stage 2: Soft Orthogonality")
            print("=" * 70)
        # Reset momentum when switching stages because W may change sharply.
        if self.riem_optimizer is not None:
            self.riem_optimizer.reset()
            # ─────────────────────────────────────────────────────────

        if self.use_tv and not self.tv_manager.is_initialized():
            self.tv_manager.update(self.W_backend)

        for epoch in range(epochs):
            if self.use_tv and epoch > 0 and epoch % self.tv_update_freq == 0:
                self.tv_manager.update(self.W_backend)

            self._update_W(use_tv_now=self.use_tv)
            self._update_Q_adam(X, epoch=epoch, stage=2)   # no subsample args

            if self.verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                self._log_metrics(X, epoch, stage=2)

    # ------------------------------------------------------------------  
    # W update routing  
    # ------------------------------------------------------------------  

    def _update_W(self, use_tv_now: bool):  
        """Route W update to GPU-full / GPU-chunked / CPU path."""  
        Q_backend   = self._get_Q_backend()  
        gpu_strategy = self.strategy['W_location'] == 'gpu'  

        if gpu_strategy and self.data_manager and not self.data_manager.is_on_gpu():  
            self._update_W_chunked(Q_backend, use_tv_now)  
        elif gpu_strategy:  
            self._update_W_full_gpu(Q_backend, use_tv_now)  
        else:  
            self._update_W_cpu(use_tv_now)  

        if gpu_strategy:  
            self.W = cp.asnumpy(self.W_backend)  

    def _get_Q_backend(self):  
        Q_np = self.Q_torch.detach().cpu().numpy() if self.Q_torch is not None else self.Q  
        return cp.asarray(Q_np, dtype=cp.float32) if self.strategy['W_location'] == 'gpu' else Q_np  

    def _update_W_full_gpu(self, Q_backend, use_tv_now):  
        self.W_backend = update_W_multiplicative(  
            self.W_backend, self.X_backend, Q_backend,  
            L=None if use_tv_now else self.L_backend,  
            alpha=self.alpha,  
            tv_manager=self.tv_manager if use_tv_now else None,  
            lambda_l1=self.lambda_l1_W_backend,  
            backend='cupy',  
            data_manager=self.data_manager,  
        )  

    def _update_W_chunked(self, Q_backend, use_tv_now):  
        from .core.sparse_utils import sparse_dense_matmul_chunked_gpu  
        Q_np   = cp.asnumpy(Q_backend) if isinstance(Q_backend, cp.ndarray) else Q_backend  
        XQ_gpu = sparse_dense_matmul_chunked_gpu(  
            self.X_backend, cp.asarray(Q_np, dtype=cp.float32),  
            chunk_size='auto', transpose_X=False, verbose=False,  
        )  
        if use_tv_now:  
            self.W_backend = update_W_hybrid_manual_tv(  
                XQ_gpu, Q_backend, self.W_backend,  
                self.tv_manager, self.alpha, self.lambda_l1_W_backend,  
            )  
        else:  
            self.W_backend = update_W_hybrid_manual(  
                XQ_gpu, Q_backend, self.W_backend,  
                self.L_backend, self.alpha, self.lambda_l1_W_backend,  
            )  

    def _update_W_cpu(self, use_tv_now):  
        self.W = update_W_multiplicative(  
            self.W, self.X_backend, self.Q,  
            L=None if use_tv_now else self.L_backend,  
            alpha=self.alpha,  
            tv_manager=self.tv_manager if use_tv_now else None,  
            lambda_l1=self.lambda_l1_W_backend,  
            backend='numpy',  
            data_manager=None,  
        )  
        self.W_backend = self.W  

    # ------------------------------------------------------------------  
    # Q updates  
    # ------------------------------------------------------------------  

    def _update_Q_procrustes(self, X, use_subsample, subsample_size):  
        """Update Q using closed-form Procrustes."""  
        X_is_sparse = issparse(X) or (CUPY_AVAILABLE and hasattr(X, 'toarray'))  
        if X_is_sparse and (not use_subsample or subsample_size is None or subsample_size >= X.shape[0]):  
            subsample_size = min(15000, X.shape[0] // 2)  
            use_subsample  = True  

        if use_subsample and subsample_size is not None and subsample_size < X.shape[0]:  
            idx   = np.random.choice(X.shape[0], size=min(subsample_size, X.shape[0]), replace=False)  
            W_sub = self.W[idx]  
            X_sub = X[idx]  
        else:  
            W_sub, X_sub = self.W, X  

        if issparse(X_sub) or (CUPY_AVAILABLE and hasattr(X_sub, 'toarray')):  
            X_sub = X_sub.toarray()  
        if X_sub.dtype != np.float32:  
            X_sub = X_sub.astype(np.float32)  

        q_0  = update_q0_closed_form(W_sub[:, 0], W_sub[:, 1:], self.Q[:, 1:], X_sub)  
        Q_s  = update_Qs_procrustes(W_sub[:, 1:], self.Q[:, 1:], X_sub, q_0, W_sub[:, 0])  
        self.Q = np.column_stack([q_0, Q_s])  

        # Sync back to Q_torch in-place  
        self.Q_torch.data.copy_(  
            torch.tensor(self.Q, dtype=torch.float32, device=self.Q_torch.device)  
        )  

    def _update_Q_adam(self, X, epoch: int = 0, stage: int = 2):
        """
        Update Q via Adam. Subsampling removed: WtW/WtX computed from full X
        using scipy sparse SpMM on CPU (exploits ~95% sparsity of X).
        The Adam loop only operates on (p,r) and (r,r) tensors on GPU.
        """
        if self.Q_torch is None:
            raise RuntimeError("Q_torch not initialised — call _setup_backend() first.")
        
        diag = self.verbose and (epoch % 10 == 0)

        # Retain optimizer state across Q updates within the same stage.
        self.Q_torch, self.riem_optimizer = update_Q_adam(
            self.Q_torch,
            self.W,
            X,
            beta             = self.beta,  # Accepted but ignored by the Q updater.
            lambda_l1        = self.lambda_l1_Q,
            riem_optimizer   = self.riem_optimizer,
            lr               = None,
            n_steps          = self.adam_steps,
            background_col   = 0,
            ortho_mode       = self.ortho_mode,
            huber_delta      = self.huber_delta,
            lambda_neg       = self.lambda_neg,
            smooth_l1_delta  = self.smooth_l1_delta,
            grad_clip_norm   = self.grad_clip_norm,
            # Diagnostic output is enabled only at selected epochs.
            verbose_diag     = diag,
            diag_prefix      = f'ep={epoch:03d} s{stage}',
        )

        self.Q = self.Q_torch.detach().cpu().numpy()

        if TORCH_AVAILABLE and torch.cuda.is_available():  
            torch.cuda.empty_cache()  

    # ------------------------------------------------------------------  
    # Logging / diagnostics  
    # ------------------------------------------------------------------  

    def _log_metrics(self, X, epoch, stage):
        sample_size = 10000 if X.shape[0] > 50000 else None
        metrics = {'recon_err': compute_relative_error(X, self.W, self.Q, sample_size)}

        Q_s = self.Q[:, 1:]
        r_s = Q_s.shape[1]

        # Report signal-column orthogonality.
        gram     = Q_s.T @ Q_s
        ortho_abs = np.linalg.norm(gram - np.eye(r_s), 'fro')

        # Normalised: independent of r_s, target < 0.1
        metrics['ortho'] = ortho_abs / np.sqrt(r_s)

        # Max pairwise cosine: most interpretable, target < 0.1
        norms          = np.linalg.norm(Q_s, axis=0, keepdims=True) + 1e-10
        gram_cos       = (Q_s / norms).T @ (Q_s / norms)
        np.fill_diagonal(gram_cos, 0.0)
        metrics['max_cos'] = float(np.abs(gram_cos).max())
        # ─────────────────────────────────────────────────────────────

        if epoch % 20 == 0 and self.lambda_l1_W is not None:
            metrics['W_sparse'] = f"{np.mean(np.abs(self.W)  < 1e-3):.1%}"
            metrics['Q_sparse'] = f"{np.mean(np.abs(Q_s) < 1e-3):.1%}"

        log_progress(epoch, metrics, stage=stage, interval=10)

    def _print_final_statistics(self, X):  
        print("\nFinal Statistics:")  
        print("-" * 70)  
        print(f"Reconstruction error:    {compute_relative_error(X, self.W, self.Q):.4f}")  

        Q_s      = self.Q[:, 1:]  
        ortho    = np.linalg.norm(Q_s.T @ Q_s - np.eye(self.r - 1), 'fro')  
        print(f"Orthogonality violation: {ortho:.4f}")  

        if self.lambda_l1_W is not None:  
            print(f"W sparsity: {np.mean(np.abs(self.W)  < 1e-3):.1%}")  
            print(f"Q sparsity: {np.mean(np.abs(Q_s) < 1e-3):.1%}")  

        W_norms, Q_norms = np.linalg.norm(self.W, axis=0), np.linalg.norm(self.Q, axis=0)  
        print(f"\nComponent scales:")  
        print(f"  W col-norms: [{W_norms.min():.2f}, {W_norms.max():.2f}]")  
        print(f"  Q col-norms: [{Q_norms.min():.2f}, {Q_norms.max():.2f}]")  

        if self.use_tv and self.tv_manager.is_initialized():  
            w_tilde, _, _ = self.tv_manager.get_regularization_terms(self.W_backend)  
            w_static      = self.tv_manager.w_static  
            if CUPY_AVAILABLE and isinstance(w_tilde, cp.ndarray):  
                w_tilde  = cp.asnumpy(w_tilde)  
                w_static = cp.asnumpy(w_static)  
            print("\nTV Statistics:")  
            validate_tv_weights(w_tilde, w_static, verbose=True)  

    # ------------------------------------------------------------------  
    # Cleanup / accessors  
    # ------------------------------------------------------------------  

    def _finalize(self):  
        """Sync results to NumPy and free GPU memory."""  
        if self.strategy and self.strategy['W_location'] == 'gpu' and isinstance(self.W_backend, cp.ndarray):  
            self.W = cp.asnumpy(self.W_backend)  

        if self.Q_torch is not None:  
            self.Q = self.Q_torch.detach().cpu().numpy()  

        if self.data_manager:  
            self.data_manager.free_gpu_memory()  

        if self.strategy and self.strategy['W_location'] == 'gpu':  
            del self.W_backend, self.L_backend  
            if CUPY_AVAILABLE:  
                cp.get_default_memory_pool().free_all_blocks()  

        if self.Q_torch is not None:  
            del self.Q_torch, self.riem_optimizer  
            if TORCH_AVAILABLE and torch.cuda.is_available():  
                torch.cuda.empty_cache()  

    def get_background(self):  
        if self.W is None or self.Q is None:  
            raise ValueError("Model not fitted")  
        return self.W[:, 0].copy(), self.Q[:, 0].copy()  

    def get_signals(self):  
        if self.W is None or self.Q is None:  
            raise ValueError("Model not fitted")  
        return self.W[:, 1:].copy(), self.Q[:, 1:].copy()  


# ============================================================================  
# Helper: ensure X is sparse CSR float32  
# ============================================================================  

def _ensure_sparse_csr_float32(  
    X: Union[np.ndarray, spmatrix],  
    verbose: bool = False,  
) -> csr_matrix:  
    """Convert X to scipy CSR float32, warning if a dense array is passed."""  
    if issparse(X):  
        X = X.tocsr() if not isinstance(X, csr_matrix) else X  
        return X.astype(np.float32) if X.dtype != np.float32 else X  

    if isinstance(X, np.ndarray):  
        import warnings  
        warnings.warn(  
            f"adata.X is a dense array (shape {X.shape}, dtype {X.dtype}). "  
            "Converting to CSR sparse float32.",  
            UserWarning, stacklevel=3,  
        )  
        if verbose:  
            print(f"  [decompose] dense→sparse: shape={X.shape}, dtype→float32 CSR")  
        return csr_matrix(X.astype(np.float32))  

    raise TypeError(f"X must be ndarray or sparse matrix, got {type(X)}.")  


# ============================================================================  
# High-level API  
# ============================================================================  

def decompose(  
    adata: AnnData,  
    n_components: int = 10,  
    layer: Optional[str] = None,  
    use_highly_variable: bool = True,  
    # ── core ────────────────────────────────────────────────────────────────  
    alpha: float = 0.1,  
    beta: float = 0.05,  # Deprecated; retained for older callers.
    # ── regularisation ──────────────────────────────────────────────────────  
    lambda_l1_W: float = 0.01,  
    lambda_l1_Q: float = 80.0,  
    l1_weight_strategy: Literal['fixed', 'adaptive', 'none'] = 'adaptive',  
    lambda_neg: float = 1.0,  
    # ── spatial ─────────────────────────────────────────────────────────────  
    use_tv: bool = False,
    tv_epsilon: float = 1e-6,
    tv_update_freq: int = 5,
    tv_stage: Literal['stage2', 'both'] = 'stage2',
    # ── training ────────────────────────────────────────────────────────────
    stage1_epochs: int = 50,
    stage2_epochs: int = 100,
    # ── advanced optimisation (rarely need changing) ─────────────────────────
    ortho_mode: str = 'huber',
    huber_delta: float = 0.5,
    smooth_l1_delta: float = 0.1,
    adam_steps: int = 40,
    grad_clip_norm: float = 1.0,
    # ── runtime ─────────────────────────────────────────────────────────────
    device: Literal['auto', 'cuda', 'cpu', 'numpy'] = 'auto',
    random_state: Optional[int] = None,
    # ── initialisation ──────────────────────────────────────────────────────
    W_init: Optional[np.ndarray] = None,
    Q_init: Optional[np.ndarray] = None,
    auto_init: bool = True,
    init_kwargs: Optional[dict] = None,
    # ── output ──────────────────────────────────────────────────────────────
    copy: bool = False,
    verbose: Optional[bool] = None,
) -> Optional[AnnData]:
    """
    Decompose spatial transcriptomics data using SORT.

    Parameters
    ----------
    adata             : AnnData with spatial coordinates
    n_components      : total components (1 background + n-1 signals)
    layer             : which layer to use (default: adata.X)
    use_highly_variable : restrict to HVGs when available
    alpha             : spatial regularisation strength
    beta              : deprecated compatibility argument; ignored
    lambda_l1_W       : L1 strength for W
    lambda_l1_Q       : L1 strength for Q  (0 = disabled)
    l1_weight_strategy: 'fixed' | 'adaptive' | 'none'
    lambda_neg        : non-negativity penalty relative weight
    use_tv            : enable Total Variation regularisation
    tv_epsilon        : TV smoothing parameter
    tv_update_freq    : TV weight refresh interval (epochs)
    tv_stage          : 'stage2' | 'both'
    stage1_epochs     : epochs for hard-orthogonality stage
    stage2_epochs     : epochs for the hard-constrained Riemannian Adam stage
    ortho_mode        : 'huber'|'frobenius'|'frobenius_sq'|'l1'
    huber_delta       : Huber δ for orthogonality penalty
    smooth_l1_delta   : Huber δ for smooth-L1 on Q
    adam_steps        : Adam steps per epoch
    grad_clip_norm    : gradient clipping max-norm
    device            : 'auto' | 'cuda' | 'cpu'
    random_state      : RNG seed
    W_init / Q_init   : optional warm-start matrices
    auto_init         : use SymNMF initialisation
    init_kwargs       : extra kwargs forwarded to initialize_W_symnmf
    copy              : return a modified copy of adata
    verbose           : print progress (default: from sort.settings)

    Returns
    -------
    AnnData (copy=True) or None (in-place)
    """
    if copy:
        adata = adata.copy()

    if verbose is None:
        from .settings import settings
        verbose = settings.verbosity > 0

    if 'spatial_laplacian' not in adata.uns:
        raise ValueError(
            "Spatial Laplacian not found. Run:\n"
            "  >>> sort.build_spatial_graph(adata)\n"
            "  >>> sort.compute_laplacian(adata)"
        )

    # ── data selection ───────────────────────────────────────────────────────
    X = adata.layers[layer] if layer else adata.X

    if use_highly_variable and 'highly_variable' in adata.var:
        gene_mask  = adata.var['highly_variable'].values
        X_filtered = X[:, gene_mask]
        if verbose:
            print(f"Using {gene_mask.sum()} / {adata.n_vars} HVGs")
    else:
        gene_mask  = np.ones(adata.n_vars, dtype=bool)
        X_filtered = X

    # Normalise to CSR float32 regardless of upstream format
    X_filtered = _ensure_sparse_csr_float32(X_filtered, verbose=verbose)

    L = adata.uns['spatial_laplacian']

    # ── init kwargs ──────────────────────────────────────────────────────────
    if init_kwargs is None:
        init_kwargs = {}
    if random_state is not None and 'random_state' not in init_kwargs:
        init_kwargs['random_state'] = random_state

    # ── build and fit model ──────────────────────────────────────────────────
    model = SpatialSemiNMF(
        n_components     = n_components,
        alpha            = alpha,
        beta             = beta,
        lambda_l1_W      = lambda_l1_W,
        lambda_l1_Q      = lambda_l1_Q,
        l1_weight_strategy = l1_weight_strategy,
        lambda_neg       = lambda_neg,
        use_tv           = use_tv,
        tv_epsilon       = tv_epsilon,
        tv_update_freq   = tv_update_freq,
        tv_stage         = tv_stage,
        ortho_mode       = ortho_mode,
        huber_delta      = huber_delta,
        smooth_l1_delta  = smooth_l1_delta,
        adam_steps       = adam_steps,
        grad_clip_norm   = grad_clip_norm,
        device           = device,
        verbose          = verbose,
    )

    model.fit(
        X_filtered, L,
        W_init        = W_init,
        Q_init        = Q_init,
        stage1_epochs = stage1_epochs,
        stage2_epochs = stage2_epochs,
        auto_init     = auto_init,
        init_kwargs   = init_kwargs,
    )

    # ── store results ────────────────────────────────────────────────────────
    W        = model.W
    Q_filtered = model.Q

    # Expand Q back to full gene space (zeros for non-HVGs)
    Q_full            = np.zeros((adata.n_vars, n_components), dtype=np.float32)
    Q_full[gene_mask] = Q_filtered

    adata.obsm['X_sort']             = W
    adata.varm['sort_signatures']    = Q_full
    adata.layers['sort_reconstructed'] = (W @ Q_filtered.T).astype(np.float32)

    comp_names = ['Background'] + [f'Signal_{i}' for i in range(1, n_components)]

    adata.uns['sort'] = {
        'params': {
            'n_components'      : n_components,
            'alpha'             : alpha,
            'beta'              : beta,
            'lambda_l1_W'       : lambda_l1_W,
            'lambda_l1_Q'       : lambda_l1_Q,
            'l1_weight_strategy': l1_weight_strategy,
            'lambda_neg'        : lambda_neg,
            'use_tv'            : use_tv,
            'stage1_epochs'     : stage1_epochs,
            'stage2_epochs'     : stage2_epochs,
        },
        'gene_mask'     : gene_mask,
        'n_genes_used'  : int(gene_mask.sum()),
        'component_names': comp_names,
    }

    if verbose:
        print("\nResults stored in:")
        print("  adata.obsm['X_sort']")
        print("  adata.varm['sort_signatures']")
        print("  adata.layers['sort_reconstructed']")

    return adata if copy else None
