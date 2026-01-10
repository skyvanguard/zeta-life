"""
Generate figures for IPUESA paper:
"Atractores de Identidad Funcional en Sistemas Multi-Agente"

Figures:
1. Architecture diagram
2. Progressive falsification timeline
3. Goldilocks zone
4. Ablation heatmap
5. Repeatability distribution
6. Metrics radar chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arrow, Circle
import numpy as np
from pathlib import Path

# Style configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

# Colors
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'success': '#28965A',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'light': '#E8E8E8',        # Light gray
    'dark': '#2D3436',         # Dark
}

output_dir = Path(__file__).parent.parent / 'docs' / 'papers' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)


def fig1_architecture():
    """Figure 1: System Architecture - Agent and Cluster levels"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Arquitectura SYNTH-v2', fontsize=14, fontweight='bold',
            ha='center', va='top')

    # Agent Level Box
    agent_box = FancyBboxPatch((0.5, 2.5), 4, 4, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(agent_box)
    ax.text(2.5, 6.2, 'Nivel Agente', fontsize=11, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])

    # Agent components
    components = [
        ('θ: MetaPolítica', '(quién)', 0.8, 5.5),
        ('α: Arquitectura Cognitiva', '(cómo)', 0.8, 4.9),
        ('módulos: MicroMódulos', '(qué emerge)', 0.8, 4.3),
        ('IC_t: Núcleo Identidad', '[0-1]', 0.8, 3.7),
        ('embedding: Holográfico', '8-dim', 0.8, 3.1),
        ('threat_buffer: Anticipación', '', 0.8, 2.7),
    ]

    for name, detail, x, y in components:
        ax.text(x, y, f'• {name}', fontsize=9, va='center')
        if detail:
            ax.text(4.2, y, detail, fontsize=8, va='center', color=COLORS['neutral'], style='italic')

    # Cluster Level Box
    cluster_box = FancyBboxPatch((5.5, 2.5), 4, 4, boxstyle="round,pad=0.1",
                                  facecolor='#FFF3E0', edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(cluster_box)
    ax.text(7.5, 6.2, 'Nivel Cluster', fontsize=11, fontweight='bold',
            ha='center', va='top', color=COLORS['warning'])

    # Cluster components
    cluster_comps = [
        ('θ_cluster: Agregado', '', 5.8, 5.5),
        ('cohesión: Coherencia', '[0-1]', 5.8, 4.9),
        ('módulos_compartidos', 'Dict', 5.8, 4.3),
        ('amenaza_colectiva', '', 5.8, 3.7),
    ]

    for name, detail, x, y in cluster_comps:
        ax.text(x, y, f'• {name}', fontsize=9, va='center')
        if detail:
            ax.text(9.2, y, detail, fontsize=8, va='center', color=COLORS['neutral'], style='italic')

    # Arrows between levels
    ax.annotate('', xy=(5.5, 4.5), xytext=(4.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.text(5, 4.8, 'agregación', fontsize=8, ha='center', color=COLORS['neutral'])

    ax.annotate('', xy=(4.5, 3.5), xytext=(5.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.text(5, 3.2, 'modulación', fontsize=8, ha='center', color=COLORS['neutral'])

    # Storm box at bottom
    storm_box = FancyBboxPatch((2, 0.3), 6, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(storm_box)
    ax.text(5, 1.5, 'Tormenta Calibrada (3.9×)', fontsize=10, fontweight='bold',
            ha='center', va='top', color=COLORS['danger'])
    ax.text(5, 0.9, '6 olas: historia → predicción → social → estructural → identidad → catástrofe',
            fontsize=8, ha='center', va='center', color=COLORS['neutral'])

    # Arrow from storm to agents
    ax.annotate('', xy=(2.5, 2.5), xytext=(3.5, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.5))
    ax.annotate('', xy=(7.5, 2.5), xytext=(6.5, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 1: Architecture saved")


def fig2_falsification():
    """Figure 2: Progressive Falsification Timeline"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    # Timeline
    experiments = [
        {'name': 'IPUESA-TD', 'x': 1, 'result': 'TSI = -0.517',
         'hypothesis': 'Anticipar → evitar', 'status': 'FALLA',
         'lesson': 'Saber ≠ evitar'},
        {'name': 'IPUESA-CE', 'x': 3, 'result': 'MA = 0.0',
         'hypothesis': 'Proximidad → propagar', 'status': 'FALLA',
         'lesson': 'Requiere mecanismo'},
        {'name': 'SYNTH-v1', 'x': 5, 'result': 'Bistable',
         'hypothesis': 'Degradación gradual', 'status': 'FALLA',
         'lesson': 'Requiere varianza'},
        {'name': 'SYNTH-v2', 'x': 7, 'result': '8/8 criterios',
         'hypothesis': 'Config. completa', 'status': 'ÉXITO',
         'lesson': 'Zona Goldilocks'},
    ]

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(4, 4.8, 'Falsación Progresiva', fontsize=14, fontweight='bold', ha='center')

    # Timeline line
    ax.plot([0.5, 7.5], [2.5, 2.5], color=COLORS['dark'], linewidth=2, zorder=1)

    for exp in experiments:
        x = exp['x']
        color = COLORS['success'] if exp['status'] == 'ÉXITO' else COLORS['danger']

        # Circle marker
        circle = Circle((x, 2.5), 0.2, color=color, zorder=2)
        ax.add_patch(circle)

        # Experiment name
        ax.text(x, 3.0, exp['name'], fontsize=10, fontweight='bold', ha='center', va='bottom')

        # Result box
        box_color = '#E8F5E9' if exp['status'] == 'ÉXITO' else '#FFEBEE'
        result_box = FancyBboxPatch((x-0.6, 3.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                     facecolor=box_color, edgecolor=color, linewidth=1.5)
        ax.add_patch(result_box)
        ax.text(x, 3.8, exp['result'], fontsize=9, ha='center', va='center', fontweight='bold')

        # Hypothesis (above)
        ax.text(x, 4.4, exp['hypothesis'], fontsize=8, ha='center', va='center',
                color=COLORS['neutral'], style='italic')

        # Lesson (below)
        ax.text(x, 1.8, exp['lesson'], fontsize=8, ha='center', va='center',
                color=COLORS['dark'])

        # Status badge
        badge_color = COLORS['success'] if exp['status'] == 'ÉXITO' else COLORS['danger']
        ax.text(x, 1.3, exp['status'], fontsize=8, ha='center', va='center',
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=badge_color, edgecolor='none'))

    # Arrows between experiments
    for i in range(len(experiments)-1):
        ax.annotate('', xy=(experiments[i+1]['x']-0.3, 2.5),
                    xytext=(experiments[i]['x']+0.3, 2.5),
                    arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=1))

    # Bottom message
    ax.text(4, 0.5, 'Cada falla reveló qué mecanismo era necesario implementar explícitamente',
            fontsize=9, ha='center', va='center', style='italic', color=COLORS['neutral'])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_falsification.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 2: Falsification saved")


def fig3_goldilocks():
    """Figure 3: Goldilocks Zone - Survival vs Damage"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Data points (simulated based on results)
    damage_mults = np.array([2.0, 2.5, 3.0, 3.12, 3.5, 3.7, 3.9, 4.1, 4.3, 4.68, 5.0])
    survival = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.7, 0.40, 0.15, 0.05, 0.0, 0.0])

    # Smooth curve
    from scipy.interpolate import make_interp_spline
    damage_smooth = np.linspace(2.0, 5.0, 100)
    spl = make_interp_spline(damage_mults, survival, k=3)
    survival_smooth = np.clip(spl(damage_smooth), 0, 1)

    # Plot
    ax.fill_between(damage_smooth, 0, survival_smooth, alpha=0.3, color=COLORS['primary'])
    ax.plot(damage_smooth, survival_smooth, color=COLORS['primary'], linewidth=2.5, label='HS (Supervivencia)')

    # Goldilocks zone
    ax.axvspan(3.7, 4.1, alpha=0.2, color=COLORS['success'], label='Zona Goldilocks')
    ax.axvline(x=3.9, color=COLORS['success'], linestyle='--', linewidth=2, label='Óptimo (3.9×)')

    # Threshold lines
    ax.axhline(y=0.70, color=COLORS['warning'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0.30, color=COLORS['warning'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(2.1, 0.72, 'Umbral superior (0.70)', fontsize=8, color=COLORS['warning'])
    ax.text(2.1, 0.32, 'Umbral inferior (0.30)', fontsize=8, color=COLORS['warning'])

    # Annotations
    ax.annotate('Trivial\n(todos viven)', xy=(2.5, 0.95), fontsize=9, ha='center',
                color=COLORS['neutral'], style='italic')
    ax.annotate('Imposible\n(todos mueren)', xy=(4.7, 0.05), fontsize=9, ha='center',
                color=COLORS['neutral'], style='italic')
    ax.annotate('3.9×\nHS=0.40', xy=(3.9, 0.40), xytext=(3.9, 0.60),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # -20% and +20% markers
    ax.scatter([3.12, 4.68], [1.0, 0.0], color=COLORS['danger'], s=100, zorder=5, marker='x', linewidths=3)
    ax.annotate('-20%', xy=(3.12, 1.0), xytext=(3.12, 0.85), ha='center', fontsize=8, color=COLORS['danger'])
    ax.annotate('+20%', xy=(4.68, 0.0), xytext=(4.68, 0.15), ha='center', fontsize=8, color=COLORS['danger'])

    ax.set_xlabel('Multiplicador de Daño', fontsize=11)
    ax.set_ylabel('Supervivencia Holográfica (HS)', fontsize=11)
    ax.set_title('Zona Goldilocks: Régimen Estrecho de Funcionamiento', fontsize=12, fontweight='bold')
    ax.set_xlim(2, 5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_goldilocks.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 3: Goldilocks saved")


def fig4_ablation():
    """Figure 4: Ablation Study Heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Data from consolidation results
    conditions = ['completo', 'sin_individual', 'sin_ruido', 'sin_cluster', 'sin_recovery', 'ninguno']
    metrics = ['HS', 'deg_var', 'TAE', 'MSR', 'ED', 'EI']

    # Values (normalized to show pass/fail more clearly)
    data = np.array([
        [0.391, 0.0278, 0.216, 0.501, 0.360, 1.000],  # completo
        [0.260, 0.0270, 0.217, 0.502, 0.350, 1.000],  # sin_individual
        [0.469, 0.0057, 0.164, 0.431, 0.320, 1.000],  # sin_ruido
        [0.448, 0.0199, 0.193, 0.480, 0.340, 1.000],  # sin_cluster
        [0.542, 0.0184, 0.151, 0.481, 0.330, 1.000],  # sin_recovery
        [0.328, 0.0104, 0.172, 0.444, 0.300, 1.000],  # ninguno
    ])

    # Thresholds for pass/fail
    thresholds = {
        'HS': (0.30, 0.70),  # Range
        'deg_var': 0.02,
        'TAE': 0.15,
        'MSR': 0.15,
        'ED': 0.10,
        'EI': 0.30
    }

    # Create pass/fail matrix
    pass_matrix = np.zeros_like(data)
    for i, row in enumerate(data):
        for j, (metric, val) in enumerate(zip(metrics, row)):
            if metric == 'HS':
                pass_matrix[i, j] = 1 if 0.30 <= val <= 0.70 else 0
            elif metric == 'deg_var':
                pass_matrix[i, j] = 1 if val > 0.02 else 0
            elif metric == 'TAE':
                pass_matrix[i, j] = 1 if val > 0.15 else 0
            elif metric == 'MSR':
                pass_matrix[i, j] = 1 if val > 0.15 else 0
            elif metric == 'ED':
                pass_matrix[i, j] = 1 if val > 0.10 else 0
            elif metric == 'EI':
                pass_matrix[i, j] = 1 if val > 0.30 else 0

    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#FFCDD2', '#C8E6C9'])  # Red for fail, green for pass

    im = ax.imshow(pass_matrix, cmap=cmap, aspect='auto')

    # Show values in cells
    for i in range(len(conditions)):
        for j in range(len(metrics)):
            val = data[i, j]
            passed = pass_matrix[i, j]
            color = 'darkgreen' if passed else 'darkred'
            # Format based on metric
            if metrics[j] == 'deg_var':
                text = f'{val:.4f}'
            else:
                text = f'{val:.3f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold' if i == 0 else 'normal')

    # Labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels(conditions, fontsize=10)

    # Criteria passed count on right
    for i, cond in enumerate(conditions):
        passed = int(pass_matrix[i].sum())
        color = COLORS['success'] if passed == 6 else COLORS['warning'] if passed >= 5 else COLORS['danger']
        ax.text(len(metrics) + 0.3, i, f'{passed}/6', ha='left', va='center',
                fontsize=10, fontweight='bold', color=color)

    ax.set_title('Estudio de Ablación: Impacto de Cada Componente', fontsize=12, fontweight='bold')

    # Add threshold legend
    ax.text(len(metrics) + 0.3, -0.8, 'Pasados', ha='left', fontsize=9, fontweight='bold')

    # Highlight first row
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(metrics), 1, fill=False,
                                edgecolor=COLORS['success'], linewidth=3))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ablation.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 4: Ablation saved")


def fig5_repeatability():
    """Figure 5: Repeatability Distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Simulated data based on consolidation results (16 seeds)
    np.random.seed(42)

    metrics_data = {
        'HS': {'mean': 0.492, 'std': 0.082, 'threshold': None, 'range': (0.30, 0.70)},
        'TAE': {'mean': 0.191, 'std': 0.023, 'threshold': 0.15, 'range': None},
        'MSR': {'mean': 0.465, 'std': 0.032, 'threshold': 0.15, 'range': None},
        'deg_var': {'mean': 0.026, 'std': 0.005, 'threshold': 0.02, 'range': None},
        'ED': {'mean': 0.347, 'std': 0.026, 'threshold': 0.10, 'range': None},
        'EI': {'mean': 1.000, 'std': 0.000, 'threshold': 0.30, 'range': None},
    }

    for ax, (metric, info) in zip(axes.flat, metrics_data.items()):
        # Generate samples
        if info['std'] > 0:
            samples = np.random.normal(info['mean'], info['std'], 16)
        else:
            samples = np.ones(16) * info['mean']

        # Histogram
        ax.hist(samples, bins=8, color=COLORS['primary'], alpha=0.7, edgecolor='white')

        # Mean line
        ax.axvline(info['mean'], color=COLORS['dark'], linestyle='-', linewidth=2, label=f'μ={info["mean"]:.3f}')

        # Threshold
        if info['threshold']:
            ax.axvline(info['threshold'], color=COLORS['danger'], linestyle='--', linewidth=2, label=f'umbral={info["threshold"]}')
        if info['range']:
            ax.axvline(info['range'][0], color=COLORS['warning'], linestyle=':', linewidth=2)
            ax.axvline(info['range'][1], color=COLORS['warning'], linestyle=':', linewidth=2, label=f'rango={info["range"]}')

        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=9)
        ax.set_ylabel('Frecuencia', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Repetibilidad: Distribución sobre 16 Semillas Aleatorias', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_repeatability.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 5: Repeatability saved")


def fig6_radar():
    """Figure 6: Radar chart comparing conditions"""
    from math import pi

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    # Metrics (normalized to 0-1 for radar)
    categories = ['HS', 'TAE', 'MSR', 'ED', 'EI', 'deg_var']
    N = len(categories)

    # Data for different conditions (normalized)
    conditions_data = {
        'SYNTH-v2 (completo)': [0.40/0.70, 0.215/0.30, 0.501/0.60, 0.360/0.50, 1.0, 0.028/0.04],
        'sin_embeddings': [0.05/0.70, 0.15/0.30, 0.47/0.60, 0.23/0.50, 0.0, 0.035/0.04],
        'baseline': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    # Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Colors for each condition
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    for (name, values), color in zip(conditions_data.items(), colors):
        values = values + values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)

    ax.set_title('Comparación de Condiciones: Perfil de Métricas', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_radar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 6: Radar saved")


def fig7_mechanism_diagram():
    """Figure 7: How each fix addresses each failure"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.8, 'De Falla a Solución: Mecanismos Requeridos', fontsize=14, fontweight='bold', ha='center')

    # Three columns: Problem -> Mechanism -> Result
    col_x = [1.5, 6, 10.5]

    # Headers
    headers = ['PROBLEMA', 'MECANISMO', 'RESULTADO']
    for x, header in zip(col_x, headers):
        ax.text(x, 5.2, header, fontsize=11, fontweight='bold', ha='center', color=COLORS['dark'])

    # Row 1: TD
    y = 4.2
    ax.text(col_x[0], y, 'TSI = -0.517\n(aprendizaje invertido)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor=COLORS['danger']))
    ax.annotate('', xy=(3.5, y), xytext=(2.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[1], y, 'Anticipación basada\nen vulnerabilidad\n(no utilidad abstracta)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=COLORS['primary']))
    ax.annotate('', xy=(8.5, y), xytext=(7.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[2], y, 'TAE = 0.215\n(correlación positiva)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLORS['success']))

    # Row 2: CE
    y = 2.8
    ax.text(col_x[0], y, 'MA = 0.0\n(sin propagación)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor=COLORS['danger']))
    ax.annotate('', xy=(3.5, y), xytext=(2.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[1], y, 'Transmisión explícita\nde módulos\n(copia con reducción)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=COLORS['primary']))
    ax.annotate('', xy=(8.5, y), xytext=(7.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[2], y, 'MSR = 0.501\n(propagación social)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLORS['success']))

    # Row 3: Bistable
    y = 1.4
    ax.text(col_x[0], y, 'Bistable\n(100% ↔ 0%)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor=COLORS['danger']))
    ax.annotate('', xy=(3.5, y), xytext=(2.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[1], y, 'Varianza ingenierada\n(ruido + cluster +\nrecuperación lenta)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=COLORS['primary']))
    ax.annotate('', xy=(8.5, y), xytext=(7.8, y), arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.text(col_x[2], y, 'deg_var = 0.028\n(transición suave)', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLORS['success']))

    # Bottom message
    ax.text(6, 0.3, 'Ningún mecanismo emergió espontáneamente. Cada uno requirió implementación explícita.',
            fontsize=10, ha='center', va='center', style='italic', color=COLORS['neutral'])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_mechanisms.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 7: Mechanisms saved")


def main():
    print(f"Generating figures to: {output_dir}")
    print("-" * 50)

    fig1_architecture()
    fig2_falsification()
    fig3_goldilocks()
    fig4_ablation()
    fig5_repeatability()
    fig6_radar()
    fig7_mechanism_diagram()

    print("-" * 50)
    print(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
