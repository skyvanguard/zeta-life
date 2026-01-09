# exp_escenarios_avanzados.py
"""Experimentos avanzados: 3 escenarios de estrés para ZetaOrganism.

Escenario 1: Daño severo (80% de Fi eliminados)
Escenario 2: Múltiples daños consecutivos
Escenario 3: Competencia (Fi invasores)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism


def escenario_dano_severo():
    """Escenario 1: Eliminar 80% de los Fi."""
    print('\n' + '='*70)
    print('ESCENARIO 1: DAÑO SEVERO (80% de Fi eliminados)')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ZetaOrganism(
        grid_size=48,
        n_cells=80,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    # Cargar pesos entrenados
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
        org.cell_module.load_state_dict(weights['cell_module_state'])
        print('Pesos entrenados cargados!')
    except:
        print('Sin pesos entrenados')

    org.initialize(seed_fi=True)

    # Fase 1: Estabilización
    print('\n[FASE 1] Estabilización (100 steps)...')
    for step in range(100):
        org.step()

    pre_damage = org.get_metrics()
    print(f'Estado pre-daño: Fi={pre_damage["n_fi"]}, Coord={pre_damage["coordination"]:.3f}')

    # Fase 2: Daño SEVERO - 80% de Fi
    print('\n[FASE 2] DAÑO SEVERO: Eliminando 80% de Fi...')
    fi_cells = [c for c in org.cells if c.role_idx == 1]
    n_to_remove = int(len(fi_cells) * 0.8)

    print(f'  Fi antes: {len(fi_cells)}')
    print(f'  Eliminando: {n_to_remove} Fi (80%)')

    removed = 0
    for cell in org.cells:
        if cell.role_idx == 1 and removed < n_to_remove:
            cell.role = torch.tensor([1.0, 0.0, 0.0])
            cell.energy = 0.1
            removed += 1

    org._update_grids()
    post_damage = org.get_metrics()
    print(f'Estado post-daño: Fi={post_damage["n_fi"]}, Coord={post_damage["coordination"]:.3f}')

    # Fase 3: Regeneración
    print('\n[FASE 3] Regeneración (200 steps)...')
    regen_history = []
    for step in range(200):
        org.step()
        regen_history.append(org.get_metrics())
        if (step + 1) % 50 == 0:
            m = org.get_metrics()
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Coord={m["coordination"]:.3f}')

    post_regen = org.get_metrics()

    # Análisis
    fi_recovery = (post_regen['n_fi'] - post_damage['n_fi']) / max(pre_damage['n_fi'] - post_damage['n_fi'], 1) * 100
    print(f'\n*** RESULTADO: Fi recuperados de {post_damage["n_fi"]} a {post_regen["n_fi"]} ({fi_recovery:.1f}% recuperación) ***')

    return {
        'pre': pre_damage,
        'post': post_damage,
        'regen': post_regen,
        'history': regen_history,
        'recovery': fi_recovery
    }


def escenario_multiples_danos():
    """Escenario 2: Múltiples daños consecutivos."""
    print('\n' + '='*70)
    print('ESCENARIO 2: MÚLTIPLES DAÑOS CONSECUTIVOS')
    print('='*70)

    torch.manual_seed(43)
    np.random.seed(43)

    org = ZetaOrganism(
        grid_size=48,
        n_cells=80,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
        org.cell_module.load_state_dict(weights['cell_module_state'])
    except:
        pass

    org.initialize(seed_fi=True)

    # Estabilización inicial
    print('\n[FASE 0] Estabilización inicial (100 steps)...')
    for _ in range(100):
        org.step()

    damage_results = []
    full_history = []

    # 3 rondas de daño + recuperación
    for ronda in range(3):
        pre = org.get_metrics()
        print(f'\n--- RONDA {ronda+1} ---')
        print(f'Pre-daño: Fi={pre["n_fi"]}, Coord={pre["coordination"]:.3f}')

        # Daño: eliminar 50% de Fi actuales
        fi_cells = [c for c in org.cells if c.role_idx == 1]
        n_to_remove = max(1, len(fi_cells) // 2)

        removed = 0
        for cell in org.cells:
            if cell.role_idx == 1 and removed < n_to_remove:
                cell.role = torch.tensor([1.0, 0.0, 0.0])
                cell.energy = 0.1
                removed += 1

        org._update_grids()
        post_damage = org.get_metrics()
        print(f'Post-daño: Fi={post_damage["n_fi"]} (eliminados {n_to_remove})')

        # Recuperación (80 steps entre daños)
        for step in range(80):
            org.step()
            full_history.append(org.get_metrics())

        post_regen = org.get_metrics()
        print(f'Post-recuperación: Fi={post_regen["n_fi"]}, Coord={post_regen["coordination"]:.3f}')

        damage_results.append({
            'ronda': ronda + 1,
            'pre': pre,
            'post_damage': post_damage,
            'post_regen': post_regen,
            'eliminated': n_to_remove
        })

    # Resumen
    print('\n*** RESUMEN DE RESILIENCIA ***')
    for r in damage_results:
        recovery = r['post_regen']['n_fi'] - r['post_damage']['n_fi']
        print(f"Ronda {r['ronda']}: {r['pre']['n_fi']} -> {r['post_damage']['n_fi']} -> {r['post_regen']['n_fi']} (recupero +{recovery})")

    return {
        'rounds': damage_results,
        'history': full_history
    }


def escenario_competencia():
    """Escenario 3: Introducir Fi invasores (competencia)."""
    print('\n' + '='*70)
    print('ESCENARIO 3: COMPETENCIA (Fi INVASORES)')
    print('='*70)

    torch.manual_seed(44)
    np.random.seed(44)

    org = ZetaOrganism(
        grid_size=48,
        n_cells=80,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
        org.cell_module.load_state_dict(weights['cell_module_state'])
    except:
        pass

    org.initialize(seed_fi=True)

    # Estabilización
    print('\n[FASE 1] Estabilización (100 steps)...')
    for _ in range(100):
        org.step()

    pre_invasion = org.get_metrics()
    print(f'Pre-invasión: Fi={pre_invasion["n_fi"]}, Coord={pre_invasion["coordination"]:.3f}')

    # Identificar Fi originales (por posición)
    original_fi_positions = set()
    for cell in org.cells:
        if cell.role_idx == 1:
            original_fi_positions.add(cell.position)

    # INVASIÓN: Convertir 5 Mass alejados en Fi invasores (CORRUPT)
    print('\n[FASE 2] INVASIÓN: Introduciendo 5 Fi invasores...')

    # Encontrar masas más alejadas de los Fi originales
    mass_cells = [c for c in org.cells if c.role_idx == 0]
    fi_cells = [c for c in org.cells if c.role_idx == 1]

    if fi_cells and mass_cells:
        # Calcular distancia promedio de cada mass a todos los Fi
        mass_distances = []
        for m in mass_cells:
            avg_dist = np.mean([
                np.sqrt((m.position[0] - f.position[0])**2 +
                       (m.position[1] - f.position[1])**2)
                for f in fi_cells
            ])
            mass_distances.append((m, avg_dist))

        # Ordenar por distancia (más alejados primero)
        mass_distances.sort(key=lambda x: -x[1])

        # Convertir los 5 más alejados en CORRUPT (Fi invasores)
        invaders = []
        for i, (cell, dist) in enumerate(mass_distances[:5]):
            # Marcar como CORRUPT con alta energía
            cell.role = torch.tensor([0.0, 0.0, 1.0])  # CORRUPT
            cell.energy = 0.85
            invaders.append(cell.position)
            print(f'  Invasor {i+1} en posición {cell.position} (dist={dist:.1f})')

    org._update_grids()
    post_invasion = org.get_metrics()
    print(f'Post-invasión: Fi={post_invasion["n_fi"]}, Corrupt={post_invasion["n_corrupt"]}')

    # Fase 3: Observar competencia
    print('\n[FASE 3] Competencia (200 steps)...')
    competition_history = []

    for step in range(200):
        org.step()
        m = org.get_metrics()
        competition_history.append(m)

        if (step + 1) % 50 == 0:
            # Contar cuántos invasores sobreviven
            surviving_invaders = sum(1 for c in org.cells
                                   if c.role_idx == 2 and c.position in invaders)
            converted_to_fi = sum(1 for c in org.cells
                                 if c.role_idx == 1 and c.position in invaders)
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Corrupt={m["n_corrupt"]}, '
                  f'Invasores: {surviving_invaders} vivos, {converted_to_fi} convertidos a Fi')

    final = org.get_metrics()

    # Análisis final
    print('\n*** RESULTADO DE COMPETENCIA ***')
    print(f'Fi originales: {pre_invasion["n_fi"]}')
    print(f'Invasores iniciales: 5')
    print(f'Fi finales: {final["n_fi"]}')
    print(f'Corrupt finales: {final["n_corrupt"]}')

    # ¿Los invasores fueron absorbidos o dominaron?
    surviving_corrupt = final['n_corrupt']
    if surviving_corrupt == 0 and final['n_fi'] > pre_invasion['n_fi']:
        print('*** INTEGRACIÓN: Invasores fueron absorbidos y convertidos en Fi ***')
    elif surviving_corrupt > 0:
        print('*** COEXISTENCIA: Sistema mantiene diversidad ***')
    elif final['n_fi'] < pre_invasion['n_fi']:
        print('*** DAÑO: Invasores causaron pérdida neta de liderazgo ***')

    return {
        'pre': pre_invasion,
        'post_invasion': post_invasion,
        'final': final,
        'history': competition_history,
        'invader_positions': invaders
    }


def run_all_scenarios():
    """Ejecuta los 3 escenarios y genera visualización."""
    print('='*70)
    print('EXPERIMENTOS AVANZADOS: ZETA ORGANISM')
    print('='*70)

    # Ejecutar escenarios
    r1 = escenario_dano_severo()
    r2 = escenario_multiples_danos()
    r3 = escenario_competencia()

    # Visualización
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Escenario 1: Daño severo
    ax = axes[0, 0]
    fi_vals = [h['n_fi'] for h in r1['history']]
    ax.plot(fi_vals, 'r-', linewidth=2)
    ax.axhline(y=r1['pre']['n_fi'], color='green', linestyle='--', label=f'Pre-daño ({r1["pre"]["n_fi"]})')
    ax.axhline(y=r1['post']['n_fi'], color='orange', linestyle=':', label=f'Post-daño ({r1["post"]["n_fi"]})')
    ax.set_title(f'Escenario 1: Daño Severo (80%)\nRecuperación: {r1["recovery"]:.1f}%')
    ax.set_xlabel('Steps post-daño')
    ax.set_ylabel('Cantidad Fi')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Escenario 1: Coordinación
    ax = axes[1, 0]
    coord_vals = [h['coordination'] for h in r1['history']]
    ax.plot(coord_vals, 'g-', linewidth=2)
    ax.axhline(y=r1['pre']['coordination'], color='green', linestyle='--')
    ax.set_xlabel('Steps post-daño')
    ax.set_ylabel('Coordinación')
    ax.set_title('Homeostasis (Esc. 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Escenario 2: Múltiples daños
    ax = axes[0, 1]
    fi_vals = [h['n_fi'] for h in r2['history']]
    ax.plot(fi_vals, 'r-', linewidth=2)
    # Marcar puntos de daño
    damage_points = [0, 80, 160]
    for i, dp in enumerate(damage_points):
        if dp < len(fi_vals):
            ax.axvline(x=dp, color='black', linestyle='--', alpha=0.5)
            ax.text(dp+2, max(fi_vals)*0.9, f'Daño {i+1}', fontsize=8)
    ax.set_title('Escenario 2: Múltiples Daños\n(3 rondas de 50% eliminación)')
    ax.set_xlabel('Steps totales')
    ax.set_ylabel('Cantidad Fi')
    ax.grid(True, alpha=0.3)

    # Escenario 2: Resumen por ronda
    ax = axes[1, 1]
    rondas = [r['ronda'] for r in r2['rounds']]
    pre_vals = [r['pre']['n_fi'] for r in r2['rounds']]
    post_vals = [r['post_damage']['n_fi'] for r in r2['rounds']]
    regen_vals = [r['post_regen']['n_fi'] for r in r2['rounds']]

    x = np.arange(len(rondas))
    width = 0.25
    ax.bar(x - width, pre_vals, width, label='Pre-daño', color='green', alpha=0.7)
    ax.bar(x, post_vals, width, label='Post-daño', color='red', alpha=0.7)
    ax.bar(x + width, regen_vals, width, label='Regenerado', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ronda {r}' for r in rondas])
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Resiliencia por Ronda')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Escenario 3: Competencia
    ax = axes[0, 2]
    fi_vals = [h['n_fi'] for h in r3['history']]
    corrupt_vals = [h['n_corrupt'] for h in r3['history']]
    ax.plot(fi_vals, 'r-', linewidth=2, label='Fi')
    ax.plot(corrupt_vals, 'k--', linewidth=2, label='Corrupt (invasores)')
    ax.axhline(y=r3['pre']['n_fi'], color='green', linestyle=':', alpha=0.5, label='Fi original')
    ax.set_title('Escenario 3: Competencia\n(5 Fi invasores)')
    ax.set_xlabel('Steps post-invasión')
    ax.set_ylabel('Cantidad')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Escenario 3: Coordinación durante competencia
    ax = axes[1, 2]
    coord_vals = [h['coordination'] for h in r3['history']]
    ax.plot(coord_vals, 'g-', linewidth=2)
    ax.axhline(y=r3['pre']['coordination'], color='green', linestyle='--', label='Pre-invasión')
    ax.set_xlabel('Steps post-invasión')
    ax.set_ylabel('Coordinación')
    ax.set_title('Estabilidad durante Competencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('zeta_organism_escenarios_avanzados.png', dpi=150)
    print('\n' + '='*70)
    print('Guardado: zeta_organism_escenarios_avanzados.png')

    # Resumen final
    print('\n' + '='*70)
    print('RESUMEN DE HALLAZGOS')
    print('='*70)

    print('\n1. DAÑO SEVERO (80%):')
    print(f'   - Recuperación: {r1["recovery"]:.1f}%')
    print(f'   - Fi: {r1["post"]["n_fi"]} -> {r1["regen"]["n_fi"]}')
    print(f'   - Coord: {r1["post"]["coordination"]:.3f} -> {r1["regen"]["coordination"]:.3f}')

    print('\n2. MÚLTIPLES DAÑOS:')
    for r in r2['rounds']:
        recovery = r['post_regen']['n_fi'] - r['post_damage']['n_fi']
        print(f'   - Ronda {r["ronda"]}: +{recovery} Fi recuperados')

    print('\n3. COMPETENCIA:')
    print(f'   - Fi antes: {r3["pre"]["n_fi"]}, después: {r3["final"]["n_fi"]}')
    print(f'   - Corrupt finales: {r3["final"]["n_corrupt"]}')
    print(f'   - Coord: {r3["pre"]["coordination"]:.3f} -> {r3["final"]["coordination"]:.3f}')

    return r1, r2, r3


if __name__ == '__main__':
    run_all_scenarios()
