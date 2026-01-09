#!/usr/bin/env python
"""Test de atracción con recursos DESPLAZADOS del centro.

Diseño del experimento:
- Organismos empiezan en esquinas (10,10) y (54,54)
- Fi tiene sesgo hacia CENTRO del grid (32,32)
- Recursos en posición DESPLAZADA (50,50) - NO en el centro
- Sin feromonas: Fi va al centro, nunca encuentra recursos
- Con feromonas: Si algún Fi encuentra recursos, emite atracción
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
#DISABLED: from exp_comunicacion_quimica import ChemicalOrganism

def run_offset_foraging(name: str, pheromones_enabled: bool,
                        resource_pos: tuple = (50, 50),
                        n_steps: int = 500) -> dict:
    """Escenario con recursos desplazados."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Recursos en: {resource_pos} (NO en centro)')
    print(f'Feromonas: {"SI" if pheromones_enabled else "NO"}')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ChemicalOrganism(
        grid_size=64,
        n_cells_per_org=30,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=1,
        patch_radius=10.0,
        pheromones_enabled=pheromones_enabled,
        diffusion_rate=3.0,
        evaporation_rate=0.01,
        alarm_strength=3.0,
        attraction_strength=15.0,  # Alta atracción
        territorial_strength=0.5,
        alarm_weight=0.3,
        attraction_weight=5.0,  # Peso alto para atracción
        territorial_weight=0.2,
        enemy_detection_radius=10.0,
        directed_alarm=True,
    )

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Inicializar en esquinas
    org.initialize_foraging()

    # MOVER RECURSOS a posición desplazada (no en centro)
    org.set_patch_position(resource_pos, radius=10.0)

    initial = org.get_metrics()
    initial_in_patch = org.count_cells_in_patches()
    initial_dist_0 = org.get_distance_to_nearest_patch(0)
    initial_dist_1 = org.get_distance_to_nearest_patch(1)

    print(f'Inicial: {initial_in_patch[0]+initial_in_patch[1]} en recursos, '
          f'dist Org0={initial_dist_0:.0f}, Org1={initial_dist_1:.0f}')

    history = {
        'in_patch': [],
        'dist': [],
        'attraction': []
    }

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        if step % 25 == 0:
            in_patch = org.count_cells_in_patches()
            dist_0 = org.get_distance_to_nearest_patch(0)
            dist_1 = org.get_distance_to_nearest_patch(1)
            history['in_patch'].append(in_patch[0] + in_patch[1])
            history['dist'].append((dist_0, dist_1))
            history['attraction'].append(m['org_0_attraction'] + m['org_1_attraction'])

        if (step + 1) % 100 == 0:
            in_patch = org.count_cells_in_patches()
            dist_0 = org.get_distance_to_nearest_patch(0)
            dist_1 = org.get_distance_to_nearest_patch(1)
            attract = m['org_0_attraction'] + m['org_1_attraction']
            print(f'  Step {step+1}: {in_patch[0]+in_patch[1]} en recursos, '
                  f'dist={dist_0:.0f}/{dist_1:.0f}, attract={attract:.0f}')

    final_in_patch = org.count_cells_in_patches()
    final_dist_0 = org.get_distance_to_nearest_patch(0)
    final_dist_1 = org.get_distance_to_nearest_patch(1)

    print(f'\nFINAL: {final_in_patch[0]+final_in_patch[1]} en recursos')
    print(f'  Org0: {final_in_patch[0]} células, dist={final_dist_0:.1f}')
    print(f'  Org1: {final_in_patch[1]} células, dist={final_dist_1:.1f}')

    return {
        'final_in_patch': final_in_patch[0] + final_in_patch[1],
        'final_dist': (final_dist_0, final_dist_1),
        'history': history
    }


if __name__ == '__main__':
    print("="*70)
    print("TEST: ATRACCIÓN CON RECURSOS DESPLAZADOS")
    print("="*70)
    print("Org0 en (10,10), Org1 en (54,54)")
    print("Fi tiene sesgo hacia centro (32,32)")
    print("Recursos en (15,50) - esquina que ninguno buscaría naturalmente")
    print("="*70)

    # Recursos en esquina superior izquierda
    # Lejos de ambos organismos y del centro
    result_sin = run_offset_foraging('Sin Feromonas', False, (15, 50), 600)
    result_con = run_offset_foraging('Con Atracción', True, (15, 50), 600)

    print("\n" + "="*70)
    print("COMPARACIÓN FINAL")
    print("="*70)

    diff = result_con['final_in_patch'] - result_sin['final_in_patch']
    print(f"\nCélulas en recursos:")
    print(f"  Sin feromonas: {result_sin['final_in_patch']}")
    print(f"  Con atracción: {result_con['final_in_patch']}")
    print(f"  Diferencia: {diff:+d}")

    if diff > 0:
        print(f"\n[OK] ATRACCION FUNCIONA: +{diff} celulas llegaron al recurso")
    elif diff == 0:
        print(f"\n[-] SIN DIFERENCIA")
    else:
        print(f"\n[X] ATRACCION NEGATIVA: {diff} menos celulas")

    # Evolución temporal
    print(f"\nEvolución (cada 100 steps):")
    for i, (h_sin, h_con) in enumerate(zip(result_sin['history']['in_patch'][::4],
                                           result_con['history']['in_patch'][::4])):
        print(f"  Step {i*100}: Sin={h_sin}, Con={h_con}")
