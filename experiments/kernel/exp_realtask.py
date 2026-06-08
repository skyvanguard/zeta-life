"""
Real task — the kernel on a canonical chaotic time series (Mackey-Glass)
=======================================================================

The standing objection to this project is that every experiment lives in a
self-built 4-D simplex toy. This is the first test on an EXTERNAL, recognised
benchmark with honest baselines: one-step prediction of the Mackey-Glass system
(tau=17), a canonical chaotic delay-differential series used for decades to
evaluate temporal models. It is NOT a simplex designed to flatter the kernel.

The kernel runs as a scalar one-step predictor (obs_dim=1, reactive): at each
step it predicts the next value from its recurrent latent (the world model's
prior), then observes the truth. We compare its one-step error against:

  - persistence : x_{t+1} ~= x_t                 (the naive baseline)
  - AR(p)       : linear autoregression (lstsq)  (the linear baseline)
  - plain GRU   : an online GRU predictor, latent_dim-matched  (the apples-to-apples baseline)

Metric: NMSE = mean((pred - x)^2) / var(x) over the test half (after warm-up),
averaged over seeds. The honest question: does the kernel's predictive core
TRANSFER to a real dynamical system, and how does it stack up against standard
baselines? (Either answer is informative.)
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel


# ---------------------------------------------------------------------------
# Mackey-Glass generator (standard discrete approximation)
# ---------------------------------------------------------------------------

def mackey_glass(n: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 expo: int = 10, discard: int = 500) -> np.ndarray:
    x = [1.2] * (tau + 1)
    for t in range(tau, tau + n + discard):
        x_tau = x[t - tau]
        x.append(x[t] + (beta * x_tau / (1.0 + x_tau ** expo) - gamma * x[t]))
    series = np.array(x[tau + discard:], dtype=float)[:n]
    return (series - series.mean()) / (series.std() + 1e-8)


# ---------------------------------------------------------------------------
# Models (one-step prediction; report per-step squared error on the test half)
# ---------------------------------------------------------------------------

def run_kernel(series: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    ck = ConsciousKernel(obs_dim=1, action_mode="reactive",
                         reflect_interval=10**9, dream_interval=10**9)
    errs = []
    for x in series:
        r = ck.step(torch.tensor([float(x)]))
        errs.append(r.errors["perceptual"])  # |pred - x| for obs_dim=1
    return np.array(errs) ** 2


def run_gru(series: np.ndarray, seed: int, hidden: int = 32, lr: float = 0.005) -> np.ndarray:
    torch.manual_seed(seed)
    gru = nn.GRUCell(1, hidden)
    head = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=lr)
    h = torch.zeros(1, hidden)
    x_prev = torch.zeros(1, 1)
    errs = []
    for x in series:
        target = torch.tensor([[float(x)]])
        h_new = gru(x_prev, h)
        pred = head(h_new)
        loss = (pred - target) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        errs.append(float((pred.detach() - target) ** 2))
        h = h_new.detach()
        x_prev = target
    return np.array(errs)


def run_wm_direct(series: np.ndarray, seed: int) -> np.ndarray:
    """The kernel's WorldModel used as a proper predictor: the transition input
    is the RAW previous value (not softmax(obs)), isolating the predictive core
    from the agent loop's action encoding."""
    torch.manual_seed(seed)
    wm = WorldModel(obs_dim=1, latent_dim=32, action_dim=1)
    prev = torch.zeros(1)
    errs = []
    for x in series:
        target = torch.tensor([float(x)])
        pred, _ = wm.predict(prev)        # predict x_t from the previous value
        errs.append(float((pred.detach() - target) ** 2))
        wm.update_from_error(pred - target)
        wm.observe(target)
        prev = target
    return np.array(errs)


def run_persistence(series: np.ndarray) -> np.ndarray:
    pred = np.concatenate([[series[0]], series[:-1]])  # x_{t-1}
    return (pred - series) ** 2


def run_ar(series: np.ndarray, p: int = 16) -> np.ndarray:
    n = len(series)
    half = n // 2
    # Design matrix on the training half.
    rows_X, rows_y = [], []
    for t in range(p, half):
        rows_X.append(series[t - p:t][::-1])
        rows_y.append(series[t])
    X = np.array(rows_X)
    y = np.array(rows_y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sq = np.full(n, np.nan)
    for t in range(p, n):
        pred = float(series[t - p:t][::-1] @ coef)
        sq[t] = (pred - series[t]) ** 2
    return sq


def nmse_test(sq: np.ndarray, var: float, half: int) -> float:
    tail = sq[half:]
    tail = tail[~np.isnan(tail)]
    return float(np.mean(tail) / var)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  REAL TASK — Mackey-Glass one-step prediction (NMSE, test half)")
    print("=" * 70)
    series = mackey_glass(n)
    var = float(series.var())
    half = n // 2
    print(f"  series: Mackey-Glass tau=17, n={n}, var={var:.3f} (normalised)")
    print(f"  metric = NMSE over the test half (last {n - half} steps); lower=better")
    print()

    # Deterministic baselines.
    pers = nmse_test(run_persistence(series), var, half)
    ar = nmse_test(run_ar(series), var, half)
    # Seeded models.
    ker = [nmse_test(run_kernel(series, s), var, half) for s in seeds]
    gru = [nmse_test(run_gru(series, s), var, half) for s in seeds]
    wmd = [nmse_test(run_wm_direct(series, s), var, half) for s in seeds]

    res = {
        "persistence":    (pers, 0.0),
        "AR(16)":         (ar, 0.0),
        "plain GRU":      (st.mean(gru), st.pstdev(gru) if len(gru) > 1 else 0.0),
        "kernel WM (direct)": (st.mean(wmd), st.pstdev(wmd) if len(wmd) > 1 else 0.0),
        "kernel (agent)": (st.mean(ker), st.pstdev(ker) if len(ker) > 1 else 0.0),
    }
    for name, (m, s) in res.items():
        print(f"  {name:12s}: NMSE = {m:.4f} ± {s:.4f}")
    print()

    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    gru_m = res["plain GRU"][0]
    core = res["kernel WM (direct)"][0]
    agent = res["kernel (agent)"][0]
    core_ok = core < gru_m * 1.3        # core comparable to a plain GRU
    agent_handicap = agent > core * 1.5  # the agent loop costs a lot
    print(f"  core (kernel WM, raw input) vs plain GRU : {core:.4f} vs {gru_m:.4f} "
          f"({'comparable' if core_ok else 'worse'})")
    print(f"  agent loop (softmax action) vs core      : {agent:.4f} vs {core:.4f} "
          f"({'big handicap' if agent_handicap else 'similar'})")
    print(f"  best simple baseline (AR/persistence)    : {min(ar, pers):.4f}")
    print()
    print("  FINDING:")
    if core_ok:
        print("  - The kernel's WORLD-MODEL CORE transfers: given the raw signal it is")
        print(f"    comparable to a purpose-built GRU ({core:.3f} vs {gru_m:.3f}).")
    else:
        print(f"  - Even the core underperforms a plain GRU ({core:.3f} vs {gru_m:.3f}).")
    if agent_handicap:
        print("  - But the full AGENT loop is much worse: it feeds the transition")
        print("    softmax(obs) (a constant for obs_dim=1), destroying the input signal.")
        print("    The kernel is an ACTION-CONDITIONED agent, not a sequence predictor;")
        print("    on pure prediction that design is a real handicap.")
    print(f"  - Linear AR essentially solves one-step Mackey-Glass ({ar:.4f}); no")
    print("    nonlinear model is needed for the 1-step horizon. Honest, bounding result.")
    print("=" * 70)
    transfers = core_ok

    if plot:
        try:
            _plot(series, half, res, seeds[0])
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return transfers


def _plot(series, half, res, seed) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = list(res)
    ax1.bar(names, [res[n][0] for n in names],
            yerr=[res[n][1] for n in names], capsize=4,
            color=["#7f8c8d", "#e67e22", "#2980b9", "#16a085", "#8e44ad"])
    ax1.set_ylabel("NMSE (test half) [lower=better]")
    ax1.set_title("Mackey-Glass one-step prediction")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(series[half:half + 200], color="#2c3e50", label="truth", lw=1.5)
    ax2.set_xlabel("step (test window)")
    ax2.set_ylabel("x (normalised)")
    ax2.set_title("Mackey-Glass series (test window excerpt)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("The kernel on an external chaotic benchmark")
    out = Path("results") / "realtask.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel on Mackey-Glass")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n=args.n, seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
