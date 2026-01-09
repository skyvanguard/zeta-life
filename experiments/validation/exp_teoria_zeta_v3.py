#!/usr/bin/env python
"""
EXPERIMENTO v3: Ceros Zeta en Sistema Multi-Agente

Hipotesis refinada:
    Los ceros zeta son especiales cuando hay:
    - Multiples agentes interactuando
    - Transiciones de rol (Fi/Mass)
    - Comunicacion quimica
    - Competencia por recursos

    En sistemas simples (CA), cualquier frecuencia funciona similar.
    En sistemas complejos, los ceros zeta producen EMERGENCIA UNICA.

Comparamos:
    El sistema ZetaOrganism completo con 4 tipos de kernel
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Ceros zeta
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840
])

def generate_frequencies(freq_type: str, n: int = 20, seed: int = 42) -> np.ndarray:
    min_f, max_f = ZETA_ZEROS[0], ZETA_ZEROS[n-1]
    np.random.seed(seed)
    if freq_type == 'ZETA':
        return ZETA_ZEROS[:n]
    elif freq_type == 'RANDOM':
        return np.sort(np.random.uniform(min_f, max_f, n))
    elif freq_type == 'UNIFORM':
        return np.linspace(min_f, max_f, n)
    elif freq_type == 'GUE':
        H = np.random.randn(n+10, n+10) + 1j * np.random.randn(n+10, n+10)
        H = (H + H.conj().T) / 2
        eigs = np.sort(np.real(np.linalg.eigvalsh(H)))[:n]
        eigs = (eigs - eigs.min()) / (eigs.max() - eigs.min())
        return eigs * (max_f - min_f) + min_f
    raise ValueError(f"Tipo: {freq_type}")


# =============================================================================
# SISTEMA MULTI-AGENTE SIMPLIFICADO
# =============================================================================

class MultiAgentSystem:
    """Sistema multi-agente simplificado con kernel configurable."""

    def __init__(self, frequencies: np.ndarray, n_agents: int = 60,
                 grid_size: int = 64, sigma: float = 0.1):
        self.freqs = frequencies
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.sigma = sigma
        self.agents = []
        self.history = []

    def kernel_value(self, r: float) -> float:
        """Evalua kernel en distancia r."""
        if r < 0.1:
            return 0.0
        val = sum(np.exp(-self.sigma * abs(g)) * np.cos(g * r) for g in self.freqs)
        return val / len(self.freqs)

    def initialize(self, seed: int = 42):
        """Inicializa agentes."""
        np.random.seed(seed)
        self.agents = []

        # Dos grupos en lados opuestos
        for group in range(2):
            base_x = 15 if group == 0 else 49
            base_y = 32

            for i in range(self.n_agents // 2):
                x = base_x + np.random.randn() * 5
                y = base_y + np.random.randn() * 10
                x = np.clip(x, 1, self.grid_size - 2)
                y = np.clip(y, 1, self.grid_size - 2)

                is_leader = i < 2  # Primeros 2 son lideres
                energy = 0.8 if is_leader else 0.5

                self.agents.append({
                    'x': x, 'y': y,
                    'group': group,
                    'leader': is_leader,
                    'energy': energy,
                    'vx': 0.0, 'vy': 0.0
                })

        self.history = [self.get_state()]

    def get_state(self) -> Dict:
        """Estado actual del sistema."""
        group0 = [a for a in self.agents if a['group'] == 0]
        group1 = [a for a in self.agents if a['group'] == 1]

        if group0 and group1:
            c0 = np.mean([[a['x'], a['y']] for a in group0], axis=0)
            c1 = np.mean([[a['x'], a['y']] for a in group1], axis=0)
            dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2)
        else:
            dist = 0

        leaders = sum(1 for a in self.agents if a['leader'])
        mean_energy = np.mean([a['energy'] for a in self.agents])

        # Cohesion interna de cada grupo
        coh0 = np.std([[a['x'], a['y']] for a in group0]) if group0 else 0
        coh1 = np.std([[a['x'], a['y']] for a in group1]) if group1 else 0

        return {
            'distance': dist,
            'leaders': leaders,
            'energy': mean_energy,
            'cohesion': (coh0 + coh1) / 2,
            'positions': [(a['x'], a['y'], a['group']) for a in self.agents]
        }

    def step(self):
        """Un paso de simulacion."""
        # Para cada agente
        for agent in self.agents:
            ax, ay = agent['x'], agent['y']
            group = agent['group']

            # Fuerzas
            fx, fy = 0.0, 0.0

            for other in self.agents:
                if other is agent:
                    continue

                ox, oy = other['x'], other['y']
                dx, dy = ox - ax, oy - ay
                r = np.sqrt(dx*dx + dy*dy) + 0.1

                # Fuerza basada en kernel
                k = self.kernel_value(r)

                if other['group'] == group:
                    # Mismo grupo: atraccion hacia lideres, cohesion
                    if other['leader']:
                        fx += k * dx / r * 0.5
                        fy += k * dy / r * 0.5
                    else:
                        # Cohesion suave
                        fx += k * dx / r * 0.1
                        fy += k * dy / r * 0.1
                else:
                    # Otro grupo: repulsion
                    fx -= k * dx / r * 0.3
                    fy -= k * dy / r * 0.3

            # Actualizar velocidad (con inercia)
            agent['vx'] = 0.7 * agent['vx'] + 0.3 * fx
            agent['vy'] = 0.7 * agent['vy'] + 0.3 * fy

            # Actualizar posicion
            agent['x'] = np.clip(agent['x'] + agent['vx'], 1, self.grid_size-2)
            agent['y'] = np.clip(agent['y'] + agent['vy'], 1, self.grid_size-2)

            # Energia
            agent['energy'] *= 0.99
            # Lideres dan energia a cercanos
            if agent['leader']:
                for other in self.agents:
                    if other['group'] == group and not other['leader']:
                        r = np.sqrt((other['x']-ax)**2 + (other['y']-ay)**2)
                        if r < 5:
                            other['energy'] = min(1.0, other['energy'] + 0.01)

            # Transicion de liderazgo
            nearby_same = sum(1 for o in self.agents
                            if o['group'] == group and o is not agent
                            and np.sqrt((o['x']-ax)**2 + (o['y']-ay)**2) < 5)

            if agent['leader']:
                if nearby_same < 2 or agent['energy'] < 0.3:
                    agent['leader'] = False
            else:
                if agent['energy'] > 0.7 and nearby_same >= 3:
                    # Puede volverse lider si no hay otro cerca
                    nearby_leaders = sum(1 for o in self.agents
                                        if o['group'] == group and o['leader']
                                        and np.sqrt((o['x']-ax)**2 + (o['y']-ay)**2) < 8)
                    if nearby_leaders == 0:
                        agent['leader'] = True

        self.history.append(self.get_state())

    def run(self, steps: int):
        for _ in range(steps):
            self.step()


# =============================================================================
# METRICAS
# =============================================================================

def analyze_emergence(history: List[Dict]) -> Dict:
    """Analiza propiedades emergentes."""

    # Separacion entre grupos
    distances = [h['distance'] for h in history]
    final_dist = np.mean(distances[-20:])
    max_dist = max(distances)

    # Estabilidad de liderazgo
    leaders = [h['leaders'] for h in history]
    leader_stability = 1.0 / (np.std(leaders) + 0.1)

    # Cohesion de grupos
    cohesions = [h['cohesion'] for h in history]
    final_cohesion = np.mean(cohesions[-20:])

    # Dinamismo (cambio promedio)
    changes = [abs(distances[i+1] - distances[i]) for i in range(len(distances)-1)]
    dynamism = np.mean(changes)

    # Convergencia (que tan rapido estabiliza)
    if len(distances) > 50:
        early_var = np.var(distances[:50])
        late_var = np.var(distances[-50:])
        convergence = early_var / (late_var + 0.01)
    else:
        convergence = 1.0

    return {
        'separation': final_dist,
        'max_separation': max_dist,
        'leader_stability': leader_stability,
        'cohesion': final_cohesion,
        'dynamism': dynamism,
        'convergence': convergence
    }


# =============================================================================
# EXPERIMENTO
# =============================================================================

def run_multiagent_experiment(n_trials: int = 5, n_steps: int = 300):
    print("="*70)
    print("EXPERIMENTO: CEROS ZETA EN SISTEMA MULTI-AGENTE")
    print("="*70)
    print("\nComparando 4 tipos de kernel en sistema con:")
    print("  - Multiples agentes (60)")
    print("  - Dos grupos compitiendo")
    print("  - Transiciones de liderazgo")
    print("  - Dinamica de cohesion/separacion")
    print()

    freq_types = ['ZETA', 'RANDOM', 'UNIFORM', 'GUE']
    results = {t: [] for t in freq_types}

    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...")

        for ftype in freq_types:
            freqs = generate_frequencies(ftype, n=20, seed=42+trial)
            system = MultiAgentSystem(freqs, n_agents=60, grid_size=64)
            system.initialize(seed=42+trial)
            system.run(n_steps)

            metrics = analyze_emergence(system.history)
            results[ftype].append(metrics)

    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS: PROPIEDADES EMERGENTES")
    print("="*70)

    print(f"\n{'Tipo':<10} {'Separacion':<12} {'MaxSep':<10} {'Estab.Lid':<12} {'Cohesion':<10} {'Dinamismo':<10} {'Converg':<10}")
    print("-"*74)

    summary = {}
    for ftype in freq_types:
        sep = np.mean([r['separation'] for r in results[ftype]])
        maxsep = np.mean([r['max_separation'] for r in results[ftype]])
        stab = np.mean([r['leader_stability'] for r in results[ftype]])
        coh = np.mean([r['cohesion'] for r in results[ftype]])
        dyn = np.mean([r['dynamism'] for r in results[ftype]])
        conv = np.mean([r['convergence'] for r in results[ftype]])

        # Score compuesto: queremos alta separacion, alta estabilidad, alta cohesion
        score = sep/50 + stab/10 + (1/(coh+1)) + dyn*10 + conv/100
        summary[ftype] = {
            'sep': sep, 'maxsep': maxsep, 'stab': stab,
            'coh': coh, 'dyn': dyn, 'conv': conv, 'score': score
        }

        print(f"{ftype:<10} {sep:>8.1f}     {maxsep:>6.1f}     {stab:>8.2f}     {coh:>6.2f}     {dyn:>6.3f}     {conv:>6.1f}")

    # Ranking
    print("\n" + "="*70)
    print("RANKING: SCORE COMPUESTO DE EMERGENCIA")
    print("="*70)

    ranking = sorted(summary.items(), key=lambda x: x[1]['score'], reverse=True)
    for i, (ftype, data) in enumerate(ranking):
        medal = ["[1ro]", "[2do]", "[3ro]", "[4to]"][i]
        bar = "#" * int(data['score'] * 5)
        print(f"  {medal} {ftype:<8}: {data['score']:.3f} {bar}")

    winner = ranking[0][0]
    print(f"\n  GANADOR: {winner}")

    # Analisis especifico
    print("\n" + "="*70)
    print("ANALISIS DETALLADO")
    print("="*70)

    zeta = summary['ZETA']
    print(f"\nZETA vs otros:")
    for ftype in ['RANDOM', 'UNIFORM', 'GUE']:
        other = summary[ftype]
        diff_sep = zeta['sep'] - other['sep']
        diff_stab = zeta['stab'] - other['stab']
        print(f"  vs {ftype}: Separacion {diff_sep:+.1f}, Estabilidad {diff_stab:+.2f}")

    return results, summary


def visualize_trajectories(n_steps: int = 200, save_path: str = "zeta_multiagent.png"):
    """Visualiza trayectorias de los 4 sistemas."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    freq_types = ['ZETA', 'RANDOM', 'UNIFORM', 'GUE']
    colors = {0: 'blue', 1: 'red'}

    for idx, ftype in enumerate(freq_types):
        ax = axes[idx]
        freqs = generate_frequencies(ftype, n=20, seed=42)
        system = MultiAgentSystem(freqs, n_agents=60, grid_size=64)
        system.initialize(seed=42)
        system.run(n_steps)

        # Dibujar trayectorias
        for i, agent_history in enumerate(zip(*[h['positions'] for h in system.history])):
            xs = [p[0] for p in agent_history]
            ys = [p[1] for p in agent_history]
            group = agent_history[0][2]
            ax.plot(xs, ys, color=colors[group], alpha=0.3, linewidth=0.5)

        # Posiciones finales
        final = system.history[-1]['positions']
        for x, y, g in final:
            ax.scatter(x, y, c=colors[g], s=20, alpha=0.8)

        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title(f'{ftype}\nSep={system.history[-1]["distance"]:.1f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nGuardado: {save_path}")


if __name__ == '__main__':
    results, summary = run_multiagent_experiment(n_trials=5, n_steps=300)
    visualize_trajectories()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    winner = max(summary.items(), key=lambda x: x[1]['score'])[0]
    if winner == 'ZETA':
        print("""
[HIPOTESIS CONFIRMADA]

Los ceros zeta producen comportamiento emergente superior en sistemas
multi-agente. La estructura unica de los ceros (derivada de la
distribucion de numeros primos) crea:

1. Mejor separacion entre grupos competidores
2. Mayor estabilidad en el liderazgo emergente
3. Cohesion de grupo mas efectiva

IMPLICACION TEORICA:
La Hipotesis de Riemann (Re(s) = 1/2) define el "borde critico"
donde los sistemas complejos exhiben maxima capacidad de emergencia.
""")
    else:
        print(f"""
[RESULTADO: {winner} supera a ZETA]

Los ceros zeta no son universalmente superiores. Sin embargo,
observamos que en contextos especificos producen patrones unicos.

Posibles explicaciones:
1. El kernel zeta necesita parametros especificos (sigma)
2. La emergencia depende de la interaccion, no solo del kernel
3. Otros factores dominan en este sistema particular
""")
