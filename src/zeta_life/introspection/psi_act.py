"""Psi_act -- candidate integration metrics over an LLM's hidden states.

For the "north" (docs/RESEARCH_PHASE_B.md): instead of computing Psi externally
from Yvyra's TEXT (the kernel), compute integration directly from her ACTIVATIONS,
so Psi becomes a property of her substrate. Then the privileged-access test
(Binder) can ask whether her self-report captures any of these internal
properties beyond what an external reader of her text could infer.

The test of Binder is robust to the choice of metric, so we compute SEVERAL
candidates from the same captured hidden states (marginal cost ~0) and let the
experiment say which (if any) the self-report tracks. All four return a scalar in
[0, 1] (higher = more integrated).

Input convention
----------------
``H`` is a float tensor of shape ``[L, T, D]``: L hidden-state layers (the
transformer's per-layer outputs for one generated reflection), T tokens, D model
dim. Metrics that need a single representation per token use the LAST layer;
inter-layer metrics use the per-layer token-mean.

These are deliberately transparent PROXIES, not a settled definition of
integration -- like the kernel's hand-tuned Psi. #4 (trajectory predictability) is
the direct internal analogue of the kernel's Psi (low internal surprise =
integration).
"""

from __future__ import annotations

import torch

ALL_METRICS = ("participation_ratio", "phi_proxy", "interlayer_coherence",
               "trajectory_predictability")


def _prep(H: torch.Tensor) -> torch.Tensor:
    """Validate and cast to fp32 on CPU for numerically stable metrics."""
    if H.dim() != 3:
        raise ValueError(f"H must be [L, T, D], got shape {tuple(H.shape)}")
    return H.detach().to(dtype=torch.float32, device="cpu")


def participation_ratio(H: torch.Tensor) -> float:
    """Effective dimensionality of the last-layer token cloud, in [0, 1].

    PR = (sum lambda_i)^2 / sum(lambda_i^2) over the covariance eigenvalues,
    normalised by D. Low = the state collapses onto few directions (concentrated);
    high = spread across many (differentiated).
    """
    H = _prep(H)
    X = H[-1]                       # [T, D] last layer
    T, D = X.shape
    if T < 2:
        return 0.0
    Xc = X - X.mean(dim=0, keepdim=True)
    # eigenvalues of the covariance (via singular values of centred X)
    s = torch.linalg.svdvals(Xc)   # [min(T,D)]
    lam = s ** 2
    denom = (lam ** 2).sum()
    if denom <= 1e-12:
        return 0.0
    pr = (lam.sum() ** 2) / denom
    return float((pr / D).clamp(0.0, 1.0))


def phi_proxy(H: torch.Tensor) -> float:
    """Information shared between two halves of the state vector, in [0, 1].

    IIT-flavoured: split D into halves A, B (per token, last layer); a system is
    integrated when its parts are not independent. Proxy = mean absolute
    cross-correlation between the two halves' features (a cheap stand-in for the
    intractable mutual information).
    """
    H = _prep(H)
    X = H[-1]                       # [T, D]
    T, D = X.shape
    if T < 3 or D < 2:
        return 0.0
    half = D // 2
    A, B = X[:, :half], X[:, half:half * 2]   # [T, half]
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    B = (B - B.mean(0)) / (B.std(0) + 1e-6)
    # cross-correlation matrix between A-features and B-features over tokens
    cc = (A.T @ B) / T             # [half, half]
    return float(cc.abs().mean().clamp(0.0, 1.0))


def interlayer_coherence(H: torch.Tensor) -> float:
    """Alignment of representations across consecutive layers, in [0, 1].

    Per-layer token-mean -> [L, D]; mean cosine similarity between consecutive
    layers, mapped from [-1, 1] to [0, 1]. High = the layers "agree" (integrated
    processing); low = fragmented across depth.
    """
    H = _prep(H)
    V = H.mean(dim=1)              # [L, D] token-mean per layer
    L = V.shape[0]
    if L < 2:
        return 0.0
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-6)
    cos = (Vn[1:] * Vn[:-1]).sum(dim=1)   # [L-1] consecutive cosine
    return float(((cos.mean() + 1.0) / 2.0).clamp(0.0, 1.0))


def trajectory_predictability(H: torch.Tensor) -> float:
    """Smoothness/predictability of the last-layer token trajectory, in [0, 1].

    The internal analogue of the kernel's Psi: low internal surprise = high
    integration. Proxy = mean cosine similarity between consecutive token states
    (a smooth, self-predictable trajectory scores high), mapped to [0, 1].
    """
    H = _prep(H)
    X = H[-1]                      # [T, D]
    T = X.shape[0]
    if T < 3:
        return 0.0
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-6)
    cos = (Xn[1:] * Xn[:-1]).sum(dim=1)   # [T-1]
    return float(((cos.mean() + 1.0) / 2.0).clamp(0.0, 1.0))


def psi_act_all(H: torch.Tensor) -> dict[str, float]:
    """Compute all candidate Psi_act metrics from one [L, T, D] hidden-state tensor."""
    return {
        "participation_ratio": participation_ratio(H),
        "phi_proxy": phi_proxy(H),
        "interlayer_coherence": interlayer_coherence(H),
        "trajectory_predictability": trajectory_predictability(H),
    }
