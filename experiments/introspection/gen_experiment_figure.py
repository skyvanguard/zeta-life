"""Generate a figure of the trained-introspection experiment: architecture + math."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parents[2] / "results"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(13, 15))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 97.5, "zeta-life · Introspección entrenada  P(IK)", ha="center",
        fontsize=19, fontweight="bold")
ax.text(50, 94.6, "¿Puede Qwen3-8B aprender a percibir su propio estado epistémico?",
        ha="center", fontsize=12, style="italic", color="#444")


def box(x, y, w, h, text, fc, ec="#333", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color=color))


# --- architecture ---
box(34, 87, 32, 4.6, "Pregunta MMLU  q   (+ opciones A/B/C/D)", "#e8eef7", bold=True)
arrow(50, 87, 50, 84.2)
box(33, 78.5, 34, 5.4, "Qwen3-8B  (8B params, 4-bit QLoRA)\nun forward pass", "#d4e2f4", bold=True)

arrow(42, 78.5, 26, 73.7)
arrow(58, 78.5, 74, 73.7)
box(5, 67.5, 42, 6, "logits A/B/C/D  →  respuesta\ncorrect(q) = 1[ â = a* ] ∈ {0,1}\n(ground truth · NO textual)", "#fde9d9")
box(54, 68.5, 42, 5, "hidden states  h_ℓ(q) ∈ ℝ⁴⁰⁹⁶\n(estado interno · capa ℓ = 18)", "#e2f0d9")

arrow(74, 68.5, 62, 60)
arrow(80, 68.5, 88, 60)
box(43, 54.5, 34, 5, "Probe lineal (F2.2)\nŝ = σ(w·h_ℓ + b)  →  AUROC 0.717", "#e2f0d9")
box(81, 54.5, 17, 5, "Ψ_act  (4 métricas)\nconexión zeta", "#eae3f5")

arrow(26, 67.5, 48, 47.5)   # ground truth -> training target
arrow(60, 54.5, 50, 47.5)   # state -> M1
box(27, 41.5, 46, 5.4, "LoRA  M₁  (F2.3): aprende a auto-reportar\nP_YES(q) = softmax(z_YES , z_NO)_YES", "#d4e2f4", bold=True)

arrow(50, 41.5, 50, 35.6)
box(18, 29.5, 64, 5.6,
    "Binder (F2.4):    M₁ (auto-reporte)   vs   M₂ (solo texto)\n"
    "introspección  ⟺  AUROC(M₁) > AUROC(M₂)", "#f4d4d4", bold=True)

# --- math panel ---
ax.add_patch(FancyBboxPatch((4, 2.5), 92, 23.5, boxstyle="round,pad=0.6,rounding_size=1.5",
                            fc="#fafafa", ec="#999", lw=1.3))
ax.text(50, 24.3, "Las matemáticas", ha="center", fontsize=14, fontweight="bold")

eqs = [
    "1.  Ground truth P(IK):   correct(q) = 𝟙[ argmaxₐ softmax(zₐ | q) = a* ] ∈ {0,1}",
    "2.  Probe (la señal existe):   ŝ = σ(wᵀ h_ℓ + b);   AUROC(ŝ , correct) = 0.717   (> 0.5)",
    "3.  LoRA (M₁ aprende):   W′ = W + (α/r)·B·A ,   B ∈ ℝ^(d×r), A ∈ ℝ^(r×k),   r = 16, α = 32",
    "4.  Auto-reporte:   P_YES(q) = softmax( [ z_YES , z_NO ] )₁    (answer-independent)",
    "5.  Binder (¿genuino?):   Δ = AUROC(M₁) − AUROC(M₂) ,   M₂ = clf( emb(q) )",
    "         Δ > 0  ⟹  introspección entrenada (acceso privilegiado);    Δ ≈ 0  ⟹  confabulación",
    "6.  Ψ_act (zeta):   PR = (Σ λᵢ)² / ( D · Σ λᵢ² ) ,   coh = mean cos(v_ℓ , v_ℓ₊₁) ,   λᵢ = eig cov(h)",
]
y = 21.3
for e in eqs:
    ax.text(6.5, y, e, ha="left", va="center", fontsize=11.0)
    y -= 2.95

fig.savefig(RESULTS / "experiment_arch.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print("saved", RESULTS / "experiment_arch.png")
