"""Experimento completo - 5 seeds, 50 epochs"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_life.core import ZetaLSTM, get_zeta_zeros
from zeta_life.core import ZetaLSTMResonantSimple

print('='*60)
print('ZetaLSTM: Original vs Resonante')
print('"No imponer, detectar"')
print('='*60)

gammas = get_zeta_zeros(15)
weights_np = np.array([np.exp(-0.1 * abs(g)) for g in gammas])

def zeta_noise(t, phase):
    return sum(w * np.cos(g * (t + phase)) for g, w in zip(gammas, weights_np)) / len(gammas)

def gen_batch(batch_size=32, seq_len=80):
    x = torch.zeros(batch_size, seq_len, 1)
    y = torch.zeros(batch_size, seq_len, 1)
    for b in range(batch_size):
        freq = np.random.uniform(0.1, 0.3)
        phase = np.random.uniform(0, 2*np.pi)
        signal = np.sin(freq * np.arange(seq_len) + phase)
        noise_phase = np.random.uniform(0, 10)
        noise = np.array([zeta_noise(t, noise_phase) for t in range(seq_len)])
        noise = noise / (np.std(noise) + 1e-8) * 0.8
        x[b, :, 0] = torch.tensor(signal + noise, dtype=torch.float32)
        y[b, :, 0] = torch.tensor(signal, dtype=torch.float32)
    return x, y

def train_eval(model, out_layer, seed, epochs=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(list(model.parameters()) + list(out_layer.parameters()), lr=2e-3)

    for _ in range(epochs):
        model.train()
        for _ in range(8):
            x, y = gen_batch()
            opt.zero_grad()
            loss = F.mse_loss(out_layer(model(x)[0]), y)
            loss.backward()
            opt.step()

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = gen_batch()
            losses.append(F.mse_loss(out_layer(model(x)[0]), y).item())
    return np.mean(losses)

seeds = [42, 123, 456, 789, 1011]
hidden = 48

v_results, o_results, r_results = [], [], []

print(f'\nEjecutando {len(seeds)} seeds...\n')

for seed in seeds:
    print(f'Seed {seed}:', end=' ')

    # Vanilla
    torch.manual_seed(seed)
    v = train_eval(nn.LSTM(1, hidden, batch_first=True), nn.Linear(hidden, 1), seed)
    v_results.append(v)

    # Original
    torch.manual_seed(seed)
    o = train_eval(ZetaLSTM(1, hidden, M=15, sigma=0.1, zeta_weight=0.4), nn.Linear(hidden, 1), seed)
    o_results.append(o)

    # Resonante
    torch.manual_seed(seed)
    r = train_eval(ZetaLSTMResonantSimple(1, hidden, M=15, sigma=0.1), nn.Linear(hidden, 1), seed)
    r_results.append(r)

    imp_o = (v - o) / v * 100
    imp_r = (v - r) / v * 100
    print(f'V={v:.5f}, O={o:.5f}({imp_o:+.1f}%), R={r:.5f}({imp_r:+.1f}%)')

# Agregados
v_m, v_s = np.mean(v_results), np.std(v_results)
o_m, o_s = np.mean(o_results), np.std(o_results)
r_m, r_s = np.mean(r_results), np.std(r_results)

imp_o_agg = (v_m - o_m) / v_m * 100
imp_r_agg = (v_m - r_m) / v_m * 100
imp_r_vs_o = (o_m - r_m) / o_m * 100

print('\n' + '='*60)
print('RESULTADOS:')
print(f'  Vanilla:  {v_m:.6f} (+/- {v_s:.6f})')
print(f'  Original: {o_m:.6f} (+/- {o_s:.6f}) [{imp_o_agg:+.2f}% vs V]')
print(f'  Resonant: {r_m:.6f} (+/- {r_s:.6f}) [{imp_r_agg:+.2f}% vs V]')
print(f'\n  Resonant vs Original: {imp_r_vs_o:+.2f}%')

# Wins
wins_o = sum(1 for v, o in zip(v_results, o_results) if o < v)
wins_r = sum(1 for v, r in zip(v_results, r_results) if r < v)
wins_r_vs_o = sum(1 for o, r in zip(o_results, r_results) if r < o)
print(f'\n  Win rates: O vs V: {wins_o}/5, R vs V: {wins_r}/5, R vs O: {wins_r_vs_o}/5')
print('='*60)

if wins_r_vs_o >= 3:
    print('\n*** RESONANTE SUPERIOR! "Detectar, no imponer" funciona. ***')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(3)
means = [v_m, o_m, r_m]
stds = [v_s, o_s, r_s]
colors = ['#3498db', '#27ae60', '#e74c3c']
labels = ['Vanilla\nLSTM', 'ZetaLSTM\nOriginal', 'ZetaLSTM\nResonante']

bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, capsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('MSE Loss')
ax.set_title(f'Comparacion ({len(seeds)} seeds)\n"No imponer, detectar"')
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, m) in enumerate(zip(bars, means)):
    if i == 0:
        txt = 'baseline'
    else:
        imp = (v_m - m) / v_m * 100
        txt = f'{imp:+.1f}%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.001,
            txt, ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('zeta_resonance_comparison.png', dpi=150)
print('\nSaved: zeta_resonance_comparison.png')
