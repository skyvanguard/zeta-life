"""
Experimento rapido: Comparacion de arquitecturas ZetaLSTM
Version optimizada con menos epochs
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
print('EXPERIMENTO RAPIDO: ZetaLSTM Original vs Resonante')
print('Principio: "No imponer, detectar"')
print('='*70)

gammas = get_zeta_zeros(15)
weights_np = np.array([np.exp(-0.1 * abs(g)) for g in gammas])

def zeta_noise(t, phase):
    return sum(w * np.cos(g * (t + phase)) for g, w in zip(gammas, weights_np)) / len(gammas)

class ZetaNoiseTask:
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

def train_eval(model_class, seed, hidden_size=48, epochs=60, is_vanilla=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_vanilla:
        model = nn.LSTM(1, hidden_size, batch_first=True)
    else:
        model = model_class(1, hidden_size, M=15, sigma=0.1)

    out_layer = nn.Linear(hidden_size, 1)
    task = ZetaNoiseTask(100, 0.8)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(out_layer.parameters()), lr=2e-3
    )

    # Train
    for _ in range(epochs):
        for _ in range(8):
            model.train()
            x, y = task.generate_batch(32)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = F.mse_loss(out_layer(out), y)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(20):
            x, y = task.generate_batch(32)
            out, _ = model(x)
            loss = F.mse_loss(out_layer(out), y)
            results.append(loss.item())

    return np.mean(results)

# Ejecutar
print('\nComparando 3 arquitecturas en 3 seeds...\n')
seeds = [42, 123, 456]

vanilla_results = []
original_results = []
resonant_results = []

for seed in seeds:
    print(f'Seed {seed}:')

    v = train_eval(None, seed, is_vanilla=True)
    vanilla_results.append(v)

    o = train_eval(ZetaLSTM, seed)
    original_results.append(o)

    r = train_eval(ZetaLSTMResonantSimple, seed)
    resonant_results.append(r)

    imp_o = (v - o) / v * 100
    imp_r = (v - r) / v * 100
    imp_r_vs_o = (o - r) / o * 100

    print(f'  Vanilla:  {v:.6f}')
    print(f'  Original: {o:.6f} ({imp_o:+.1f}% vs vanilla)')
    print(f'  Resonant: {r:.6f} ({imp_r:+.1f}% vs vanilla, {imp_r_vs_o:+.1f}% vs original)')
    print()

# Agregados
v_mean, v_std = np.mean(vanilla_results), np.std(vanilla_results)
o_mean, o_std = np.mean(original_results), np.std(original_results)
r_mean, r_std = np.mean(resonant_results), np.std(resonant_results)

imp_o_agg = (v_mean - o_mean) / v_mean * 100
imp_r_agg = (v_mean - r_mean) / v_mean * 100
imp_r_vs_o_agg = (o_mean - r_mean) / o_mean * 100

print('='*70)
print('RESULTADOS AGREGADOS (3 seeds):')
print(f'  Vanilla LSTM:     {v_mean:.6f} (+/- {v_std:.6f})')
print(f'  ZetaLSTM Original:{o_mean:.6f} (+/- {o_std:.6f})  [{imp_o_agg:+.2f}% vs vanilla]')
print(f'  ZetaLSTM Resonant:{r_mean:.6f} (+/- {r_std:.6f})  [{imp_r_agg:+.2f}% vs vanilla]')
print()
print(f'  Resonant vs Original: {imp_r_vs_o_agg:+.2f}%')
print('='*70)

# Win counts
wins_orig = sum(1 for v, o in zip(vanilla_results, original_results) if o < v)
wins_res = sum(1 for v, r in zip(vanilla_results, resonant_results) if r < v)
wins_res_vs_orig = sum(1 for o, r in zip(original_results, resonant_results) if r < o)

print(f'\nWin rates:')
print(f'  Original vs Vanilla: {wins_orig}/3')
print(f'  Resonant vs Vanilla: {wins_res}/3')
print(f'  Resonant vs Original: {wins_res_vs_orig}/3')

# Veredicto
if imp_r_agg > imp_o_agg and wins_res_vs_orig >= 2:
    print('\n*** RESONANTE SUPERIOR! El principio "detectar, no imponer" funciona. ***')
elif imp_r_agg > 0 and wins_res >= 2:
    print('\n[OK] Resonante mejora sobre vanilla consistentemente')
else:
    print('\n[--] Resultados mixtos, requiere mas investigacion')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Vanilla\nLSTM', 'ZetaLSTM\nOriginal', 'ZetaLSTM\nResonante']
means = [v_mean, o_mean, r_mean]
stds = [v_std, o_std, r_std]
colors = ['#3498db', '#27ae60', '#e74c3c']

bars = ax.bar(models, means, yerr=stds, color=colors, alpha=0.8, capsize=8, edgecolor='black')

ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Comparacion: "No imponer, detectar"\nTarea: Filtrado de ruido zeta', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Anotaciones
for i, (bar, mean) in enumerate(zip(bars, means)):
    if i == 0:
        label = 'baseline'
    else:
        imp = (v_mean - mean) / v_mean * 100
        label = f'{imp:+.1f}%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.001,
            label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('zeta_resonance_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved: zeta_resonance_comparison.png')
