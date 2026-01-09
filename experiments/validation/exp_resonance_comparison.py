"""
Experimento: Comparacion de arquitecturas ZetaLSTM

Compara:
1. Vanilla LSTM (baseline)
2. ZetaLSTM Original (modulacion fija)
3. ZetaLSTM Resonante (detecta cuando aplicar)

Hipotesis: El modelo resonante deberia:
- Mejorar sobre vanilla en datos con patron zeta
- Mejorar sobre ZetaLSTM fijo al no interferir cuando no hay patron
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_life.core import ZetaLSTM, get_zeta_zeros
from zeta_life.core import ZetaLSTMResonantSimple

print('='*70)
print('EXPERIMENTO: Comparacion de Arquitecturas ZetaLSTM')
print('Principio: "No imponer, detectar"')
print('='*70)

# Configuracion
gammas = get_zeta_zeros(15)
weights_np = np.array([np.exp(-0.1 * abs(g)) for g in gammas])

def zeta_noise(t, phase):
    """Ruido estructurado segun patron zeta."""
    return sum(w * np.cos(g * (t + phase)) for g, w in zip(gammas, weights_np)) / len(gammas)


class MixedNoiseTask:
    """
    Tarea mixta: algunas secuencias tienen ruido zeta, otras ruido gaussiano.

    Esto permite evaluar si el modelo resonante detecta correctamente
    cuando aplicar la memoria zeta.
    """
    def __init__(self, seq_length=100, noise_scale=0.8, zeta_ratio=0.5):
        self.seq_length = seq_length
        self.noise_scale = noise_scale
        self.zeta_ratio = zeta_ratio  # Proporcion de secuencias con ruido zeta

    def generate_batch(self, batch_size):
        x = torch.zeros(batch_size, self.seq_length, 1)
        y = torch.zeros(batch_size, self.seq_length, 1)
        is_zeta = torch.zeros(batch_size)  # 1 si tiene ruido zeta

        for b in range(batch_size):
            freq = np.random.uniform(0.1, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            signal = np.sin(freq * np.arange(self.seq_length) + phase)

            # Decidir tipo de ruido
            if np.random.random() < self.zeta_ratio:
                # Ruido zeta
                noise_phase = np.random.uniform(0, 10)
                noise = np.array([zeta_noise(t, noise_phase) for t in range(self.seq_length)])
                is_zeta[b] = 1
            else:
                # Ruido gaussiano
                noise = np.random.randn(self.seq_length) * 0.3

            noise = noise / (np.std(noise) + 1e-8) * self.noise_scale

            x[b, :, 0] = torch.tensor(signal + noise, dtype=torch.float32)
            y[b, :, 0] = torch.tensor(signal, dtype=torch.float32)

        return x, y, is_zeta


class ZetaOnlyTask:
    """Tarea solo con ruido zeta (para comparacion directa)."""
    def __init__(self, seq_length=100, noise_scale=0.8):
        self.seq_length = seq_length
        self.noise_scale = noise_scale

    def generate_batch(self, batch_size):
        x = torch.zeros(batch_size, self.seq_length, 1)
        y = torch.zeros(batch_size, self.seq_length, 1)

        for b in range(batch_size):
            freq = np.random.uniform(0.1, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            signal = np.sin(freq * np.arange(self.seq_length) + phase)
            noise_phase = np.random.uniform(0, 10)
            noise = np.array([zeta_noise(t, noise_phase) for t in range(self.seq_length)])
            noise = noise / (np.std(noise) + 1e-8) * self.noise_scale
            x[b, :, 0] = torch.tensor(signal + noise, dtype=torch.float32)
            y[b, :, 0] = torch.tensor(signal, dtype=torch.float32)

        return x, y


def train_model(model, out_layer, task, epochs=100, lr=2e-3):
    """Entrena un modelo."""
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(out_layer.parameters()),
        lr=lr
    )

    losses = []
    for epoch in range(epochs):
        model.train()
        out_layer.train()
        epoch_loss = 0

        for _ in range(10):
            if isinstance(task, MixedNoiseTask):
                x, y, _ = task.generate_batch(32)
            else:
                x, y = task.generate_batch(32)

            optimizer.zero_grad()
            output, _ = model(x)
            pred = out_layer(output)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / 10)

    return losses


def evaluate_model(model, out_layer, task, n_batches=30):
    """Evalua un modelo."""
    model.eval()
    out_layer.eval()

    results = []
    with torch.no_grad():
        for _ in range(n_batches):
            if isinstance(task, MixedNoiseTask):
                x, y, is_zeta = task.generate_batch(32)
            else:
                x, y = task.generate_batch(32)

            output, _ = model(x)
            pred = out_layer(output)
            loss = F.mse_loss(pred, y)
            results.append(loss.item())

    return np.mean(results), np.std(results)


def run_experiment(seed, hidden_size=48, epochs=100):
    """Ejecuta experimento con un seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {}

    # ===== TAREA 1: Solo ruido zeta =====
    task_zeta = ZetaOnlyTask(100, 0.8)

    # Vanilla LSTM
    vanilla = nn.LSTM(1, hidden_size, batch_first=True)
    vanilla_out = nn.Linear(hidden_size, 1)
    train_model(vanilla, vanilla_out, task_zeta, epochs)
    v_mean, v_std = evaluate_model(vanilla, vanilla_out, task_zeta)
    results['vanilla_zeta'] = (v_mean, v_std)

    # ZetaLSTM Original
    torch.manual_seed(seed)
    zeta_orig = ZetaLSTM(1, hidden_size, M=15, sigma=0.1, zeta_weight=0.4)
    orig_out = nn.Linear(hidden_size, 1)
    train_model(zeta_orig, orig_out, task_zeta, epochs)
    o_mean, o_std = evaluate_model(zeta_orig, orig_out, task_zeta)
    results['original_zeta'] = (o_mean, o_std)

    # ZetaLSTM Resonante
    torch.manual_seed(seed)
    zeta_res = ZetaLSTMResonantSimple(1, hidden_size, M=15, sigma=0.1)
    res_out = nn.Linear(hidden_size, 1)
    train_model(zeta_res, res_out, task_zeta, epochs)
    r_mean, r_std = evaluate_model(zeta_res, res_out, task_zeta)
    results['resonant_zeta'] = (r_mean, r_std)

    # ===== TAREA 2: Ruido mixto (50% zeta, 50% gaussiano) =====
    task_mixed = MixedNoiseTask(100, 0.8, zeta_ratio=0.5)

    # Vanilla
    torch.manual_seed(seed)
    vanilla2 = nn.LSTM(1, hidden_size, batch_first=True)
    vanilla2_out = nn.Linear(hidden_size, 1)
    train_model(vanilla2, vanilla2_out, task_mixed, epochs)
    v2_mean, v2_std = evaluate_model(vanilla2, vanilla2_out, task_mixed)
    results['vanilla_mixed'] = (v2_mean, v2_std)

    # Original
    torch.manual_seed(seed)
    zeta_orig2 = ZetaLSTM(1, hidden_size, M=15, sigma=0.1, zeta_weight=0.4)
    orig2_out = nn.Linear(hidden_size, 1)
    train_model(zeta_orig2, orig2_out, task_mixed, epochs)
    o2_mean, o2_std = evaluate_model(zeta_orig2, orig2_out, task_mixed)
    results['original_mixed'] = (o2_mean, o2_std)

    # Resonante
    torch.manual_seed(seed)
    zeta_res2 = ZetaLSTMResonantSimple(1, hidden_size, M=15, sigma=0.1)
    res2_out = nn.Linear(hidden_size, 1)
    train_model(zeta_res2, res2_out, task_mixed, epochs)
    r2_mean, r2_std = evaluate_model(zeta_res2, res2_out, task_mixed)
    results['resonant_mixed'] = (r2_mean, r2_std)

    return results


# Ejecutar experimento
print('\nEjecutando experimento con 3 seeds...\n')
seeds = [42, 123, 456]

all_results = {
    'vanilla_zeta': [], 'original_zeta': [], 'resonant_zeta': [],
    'vanilla_mixed': [], 'original_mixed': [], 'resonant_mixed': []
}

for seed in seeds:
    print(f'Seed {seed}...')
    results = run_experiment(seed, hidden_size=48, epochs=80)

    for key, (mean, std) in results.items():
        all_results[key].append(mean)

    # Mostrar resultados de este seed
    v_z = results['vanilla_zeta'][0]
    o_z = results['original_zeta'][0]
    r_z = results['resonant_zeta'][0]
    v_m = results['vanilla_mixed'][0]
    o_m = results['original_mixed'][0]
    r_m = results['resonant_mixed'][0]

    print(f'  Tarea Zeta:  Vanilla={v_z:.5f}, Original={o_z:.5f} ({(v_z-o_z)/v_z*100:+.1f}%), Resonant={r_z:.5f} ({(v_z-r_z)/v_z*100:+.1f}%)')
    print(f'  Tarea Mixta: Vanilla={v_m:.5f}, Original={o_m:.5f} ({(v_m-o_m)/v_m*100:+.1f}%), Resonant={r_m:.5f} ({(v_m-r_m)/v_m*100:+.1f}%)')
    print()


# Calcular agregados
print('='*70)
print('RESULTADOS AGREGADOS:')
print('-'*70)

def calc_stats(key):
    vals = all_results[key]
    return np.mean(vals), np.std(vals)

# Tarea Zeta
print('\nTAREA: Solo ruido zeta')
v_z_mean, v_z_std = calc_stats('vanilla_zeta')
o_z_mean, o_z_std = calc_stats('original_zeta')
r_z_mean, r_z_std = calc_stats('resonant_zeta')

print(f'  Vanilla LSTM:     {v_z_mean:.6f} (+/- {v_z_std:.6f})')
imp_o = (v_z_mean - o_z_mean) / v_z_mean * 100
imp_r = (v_z_mean - r_z_mean) / v_z_mean * 100
print(f'  ZetaLSTM Original:{o_z_mean:.6f} (+/- {o_z_std:.6f})  [{imp_o:+.2f}% vs vanilla]')
print(f'  ZetaLSTM Resonant:{r_z_mean:.6f} (+/- {r_z_std:.6f})  [{imp_r:+.2f}% vs vanilla]')

# Tarea Mixta
print('\nTAREA: Ruido mixto (50% zeta, 50% gaussiano)')
v_m_mean, v_m_std = calc_stats('vanilla_mixed')
o_m_mean, o_m_std = calc_stats('original_mixed')
r_m_mean, r_m_std = calc_stats('resonant_mixed')

print(f'  Vanilla LSTM:     {v_m_mean:.6f} (+/- {v_m_std:.6f})')
imp_o_m = (v_m_mean - o_m_mean) / v_m_mean * 100
imp_r_m = (v_m_mean - r_m_mean) / v_m_mean * 100
print(f'  ZetaLSTM Original:{o_m_mean:.6f} (+/- {o_m_std:.6f})  [{imp_o_m:+.2f}% vs vanilla]')
print(f'  ZetaLSTM Resonant:{r_m_mean:.6f} (+/- {r_m_std:.6f})  [{imp_r_m:+.2f}% vs vanilla]')

# Comparacion resonante vs original
print('\n' + '-'*70)
print('COMPARACION RESONANTE vs ORIGINAL:')
imp_res_vs_orig_zeta = (o_z_mean - r_z_mean) / o_z_mean * 100
imp_res_vs_orig_mixed = (o_m_mean - r_m_mean) / o_m_mean * 100
print(f'  Tarea Zeta:  Resonante es {imp_res_vs_orig_zeta:+.2f}% vs Original')
print(f'  Tarea Mixta: Resonante es {imp_res_vs_orig_mixed:+.2f}% vs Original')

print('='*70)

# Veredicto
if imp_r > imp_o and imp_r_m > imp_o_m:
    print('\n*** RESONANTE GANA EN AMBAS TAREAS! ***')
    print('El principio "detectar, no imponer" funciona.')
elif imp_r_m > imp_o_m:
    print('\n[OK] Resonante mejor en tarea mixta (donde importa adaptarse)')
elif imp_r > imp_o:
    print('\n[OK] Resonante mejor en tarea zeta pura')
else:
    print('\n[--] Original aun mejor. Ajustar arquitectura resonante.')


# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Tarea Zeta
models = ['Vanilla', 'Original', 'Resonant']
zeta_means = [v_z_mean, o_z_mean, r_z_mean]
zeta_stds = [v_z_std, o_z_std, r_z_std]
colors = ['blue', 'green', 'red']
axes[0].bar(models, zeta_means, yerr=zeta_stds, color=colors, alpha=0.7, capsize=5)
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Tarea: Solo Ruido Zeta')
axes[0].grid(True, alpha=0.3, axis='y')

# Tarea Mixta
mixed_means = [v_m_mean, o_m_mean, r_m_mean]
mixed_stds = [v_m_std, o_m_std, r_m_std]
axes[1].bar(models, mixed_means, yerr=mixed_stds, color=colors, alpha=0.7, capsize=5)
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Tarea: Ruido Mixto (50% zeta, 50% gaussiano)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Comparacion: Original vs Resonante\n"No imponer, detectar"', fontsize=12)
plt.tight_layout()
plt.savefig('zeta_resonance_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved: zeta_resonance_comparison.png')
