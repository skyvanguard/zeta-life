# exp_memoria_temporal.py
"""Experimento: Memoria temporal en ZetaOrganism.

Hipotesis: El sistema podria "recordar" patrones de dano anteriores,
mostrando:
1. Recuperacion mas rapida en danos repetidos (aprendizaje)
2. Reorganizacion preventiva (anticipacion)
3. Patrones de Fi diferentes post-trauma

La memoria zeta (ZetaMemoryGatedSimple) deberia capturar
dependencias temporales a largo plazo.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism


def apply_localized_damage(org, region='left', intensity=0.8):
    """Aplica dano localizado en una region especifica."""
    grid_size = org.grid_size
    damaged_count = 0

    for cell in org.cells:
        x, y = cell.position
        apply_damage = False

        if region == 'left' and x < grid_size * 0.3:
            apply_damage = True
        elif region == 'right' and x > grid_size * 0.7:
            apply_damage = True
        elif region == 'top' and y > grid_size * 0.7:
            apply_damage = True
        elif region == 'bottom' and y < grid_size * 0.3:
            apply_damage = True
        elif region == 'center':
            cx, cy = grid_size / 2, grid_size / 2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < grid_size * 0.2:
                apply_damage = True

        if apply_damage and cell.role_idx == 1:  # Solo danar Fi
            if np.random.random() < intensity:
                cell.role = torch.tensor([1.0, 0.0, 0.0])  # Convertir a Mass
                cell.energy = 0.1
                damaged_count += 1

    org._update_grids()
    return damaged_count


def measure_recovery_time(org, target_fi, max_steps=100):
    """Mide cuantos steps toma recuperar un numero objetivo de Fi."""
    history = []
    for step in range(max_steps):
        org.step()
        m = org.get_metrics()
        history.append(m)
        if m['n_fi'] >= target_fi:
            return step + 1, history
    return max_steps, history


def get_fi_distribution(org):
    """Obtiene distribucion espacial de Fi."""
    fi_positions = []
    for cell in org.cells:
        if cell.role_idx == 1:
            fi_positions.append(cell.position)
    return fi_positions


def run_memory_experiment():
    """Ejecuta experimento de memoria temporal."""
    print('='*70)
    print('EXPERIMENTO: MEMORIA TEMPORAL')
    print('='*70)
    print('Hipotesis: El sistema aprende de danos anteriores')

    torch.manual_seed(42)
    np.random.seed(42)

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
        print('Pesos entrenados cargados!')
    except:
        print('Sin pesos entrenados')

    org.initialize(seed_fi=True)

    # === FASE 0: Estabilizacion ===
    print('\n[FASE 0] Estabilizacion inicial (100 steps)...')
    for _ in range(100):
        org.step()

    baseline = org.get_metrics()
    baseline_fi_dist = get_fi_distribution(org)
    print(f'Baseline: Fi={baseline["n_fi"]}, Coord={baseline["coordination"]:.3f}')

    # Guardar historial completo
    full_history = []
    damage_events = []
    recovery_times = []

    # === EXPERIMENTO: 4 rondas de dano en la misma region ===
    damage_region = 'left'
    target_recovery = int(baseline['n_fi'] * 0.8)  # Recuperar al 80%

    print(f'\n*** Aplicando 4 rondas de dano en region "{damage_region}" ***')
    print(f'Target de recuperacion: {target_recovery} Fi')

    for ronda in range(4):
        print(f'\n--- RONDA {ronda + 1} ---')

        pre_damage = org.get_metrics()
        pre_fi_dist = get_fi_distribution(org)

        # Aplicar dano
        damaged = apply_localized_damage(org, region=damage_region, intensity=0.9)
        post_damage = org.get_metrics()

        print(f'  Pre-dano: Fi={pre_damage["n_fi"]}')
        print(f'  Fi danados: {damaged}')
        print(f'  Post-dano: Fi={post_damage["n_fi"]}')

        damage_events.append({
            'ronda': ronda + 1,
            'step': len(full_history),
            'pre_fi': pre_damage['n_fi'],
            'damaged': damaged,
            'post_fi': post_damage['n_fi']
        })

        # Medir tiempo de recuperacion
        recovery_target = min(target_recovery, pre_damage['n_fi'])
        steps_to_recover, recovery_history = measure_recovery_time(
            org, recovery_target, max_steps=80
        )

        full_history.extend(recovery_history)
        recovery_times.append(steps_to_recover)

        post_recovery = org.get_metrics()
        post_fi_dist = get_fi_distribution(org)

        print(f'  Tiempo de recuperacion: {steps_to_recover} steps')
        print(f'  Post-recuperacion: Fi={post_recovery["n_fi"]}, Coord={post_recovery["coordination"]:.3f}')

        # Analizar cambio en distribucion de Fi
        # Contar Fi en region izquierda vs derecha
        left_fi_pre = sum(1 for p in pre_fi_dist if p[0] < 24)
        left_fi_post = sum(1 for p in post_fi_dist if p[0] < 24)
        right_fi_pre = sum(1 for p in pre_fi_dist if p[0] >= 24)
        right_fi_post = sum(1 for p in post_fi_dist if p[0] >= 24)

        print(f'  Distribucion Fi: Izq {left_fi_pre}->{left_fi_post}, Der {right_fi_pre}->{right_fi_post}')

    # === FASE CONTROL: Dano en region diferente ===
    print('\n--- CONTROL: Dano en region "right" (nueva) ---')

    pre_damage = org.get_metrics()
    damaged = apply_localized_damage(org, region='right', intensity=0.9)
    post_damage = org.get_metrics()

    print(f'  Pre-dano: Fi={pre_damage["n_fi"]}')
    print(f'  Fi danados: {damaged}')
    print(f'  Post-dano: Fi={post_damage["n_fi"]}')

    recovery_target = min(target_recovery, pre_damage['n_fi'])
    control_recovery_time, control_history = measure_recovery_time(
        org, recovery_target, max_steps=80
    )
    full_history.extend(control_history)

    print(f'  Tiempo de recuperacion (control): {control_recovery_time} steps')

    # === ANALISIS ===
    print('\n' + '='*70)
    print('ANALISIS DE MEMORIA TEMPORAL')
    print('='*70)

    print(f'\n{"Ronda":<10} {"Tiempo Recuperacion":<20} {"Mejora vs Ronda 1":<20}')
    print('-'*50)
    for i, t in enumerate(recovery_times):
        if i == 0:
            mejora = "baseline"
        else:
            mejora = f'{((recovery_times[0] - t) / recovery_times[0] * 100):+.1f}%'
        print(f'{i+1:<10} {t:<20} {mejora:<20}')

    print(f'{"Control":<10} {control_recovery_time:<20} {"(nueva region)":<20}')

    # Detectar aprendizaje
    print('\n*** DETECCION DE MEMORIA ***')

    # Tendencia de tiempos de recuperacion
    if len(recovery_times) >= 2:
        trend = np.polyfit(range(len(recovery_times)), recovery_times, 1)[0]
        if trend < -0.5:
            print(f'  APRENDIZAJE DETECTADO: Tendencia negativa ({trend:.2f} steps/ronda)')
            print('  El sistema recupera mas rapido con cada dano repetido')
        elif trend > 0.5:
            print(f'  FATIGA DETECTADA: Tendencia positiva ({trend:.2f} steps/ronda)')
            print('  El sistema se recupera mas lento con danos repetidos')
        else:
            print(f'  SIN TENDENCIA CLARA: ({trend:.2f} steps/ronda)')

    # Comparar con control
    avg_trained = np.mean(recovery_times[-2:])  # Ultimas 2 rondas
    if control_recovery_time > avg_trained * 1.2:
        print(f'  ESPECIFICIDAD: Dano en nueva region toma mas tiempo ({control_recovery_time} vs {avg_trained:.1f})')
        print('  El sistema "aprende" la region especifica de dano')
    elif control_recovery_time < avg_trained * 0.8:
        print(f'  GENERALIZACION: Dano en nueva region se recupera rapido')
        print('  El aprendizaje se transfiere a otras regiones')
    else:
        print(f'  SIN DIFERENCIA SIGNIFICATIVA entre regiones')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Tiempos de recuperacion
    ax = axes[0, 0]
    rondas = list(range(1, len(recovery_times) + 1))
    bars = ax.bar(rondas, recovery_times, color='steelblue', alpha=0.7, label='Region entrenada')
    ax.bar([len(recovery_times) + 1], [control_recovery_time], color='coral', alpha=0.7, label='Region nueva')
    ax.axhline(y=np.mean(recovery_times), color='blue', linestyle='--', alpha=0.5, label='Promedio')
    ax.set_xlabel('Ronda de Dano')
    ax.set_ylabel('Steps para Recuperacion')
    ax.set_title('Tiempo de Recuperacion por Ronda')
    ax.set_xticks(list(range(1, len(recovery_times) + 2)))
    ax.set_xticklabels([str(i) for i in range(1, len(recovery_times) + 1)] + ['Ctrl'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Evolucion de Fi
    ax = axes[0, 1]
    fi_vals = [h['n_fi'] for h in full_history]
    ax.plot(fi_vals, 'r-', linewidth=1.5)
    # Marcar eventos de dano
    for event in damage_events:
        ax.axvline(x=event['step'], color='black', linestyle='--', alpha=0.5)
        ax.text(event['step'] + 2, max(fi_vals) * 0.95, f'D{event["ronda"]}', fontsize=8)
    ax.axhline(y=baseline['n_fi'], color='green', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolucion de Fi (D=Dano)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Coordinacion
    ax = axes[0, 2]
    coord_vals = [h['coordination'] for h in full_history]
    ax.plot(coord_vals, 'g-', linewidth=1.5)
    for event in damage_events:
        ax.axvline(x=event['step'], color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Coordinacion')
    ax.set_title('Estabilidad de Coordinacion')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. Curva de aprendizaje
    ax = axes[1, 0]
    ax.plot(rondas, recovery_times, 'bo-', linewidth=2, markersize=10, label='Observado')
    # Linea de tendencia
    z = np.polyfit(rondas, recovery_times, 1)
    p = np.poly1d(z)
    ax.plot(rondas, p(rondas), 'r--', linewidth=2, label=f'Tendencia ({z[0]:.2f}/ronda)')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Tiempo de Recuperacion')
    ax.set_title('Curva de Aprendizaje')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Comparacion primera vs ultima ronda
    ax = axes[1, 1]
    categories = ['Ronda 1', 'Ronda 4', 'Control']
    times = [recovery_times[0], recovery_times[-1], control_recovery_time]
    colors = ['lightcoral', 'lightgreen', 'lightskyblue']
    ax.bar(categories, times, color=colors, edgecolor='black')
    ax.set_ylabel('Steps de Recuperacion')
    ax.set_title('Comparacion: Primera vs Ultima vs Control')
    ax.grid(True, alpha=0.3)

    # 6. Estado final
    ax = axes[1, 2]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 60
        ax.scatter(x, y, c=color, s=size, alpha=0.7)
    # Marcar regiones de dano
    ax.axvline(x=48*0.3, color='orange', linestyle='--', alpha=0.7, label='Zona dano izq')
    ax.axvline(x=48*0.7, color='purple', linestyle='--', alpha=0.7, label='Zona dano der')
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_title('Estado Final')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_memoria.png', dpi=150)
    print('\nGuardado: zeta_organism_memoria.png')

    return {
        'recovery_times': recovery_times,
        'control_time': control_recovery_time,
        'damage_events': damage_events,
        'history': full_history
    }


if __name__ == '__main__':
    run_memory_experiment()
