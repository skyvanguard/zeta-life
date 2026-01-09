"""
Robust comparison with multiple seeds
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_life.core import ZetaLSTM, get_zeta_zeros

print('=' * 70)
print('ZETA LSTM - Robust Multi-Seed Comparison')
print('=' * 70)

gammas = get_zeta_zeros(15)
weights_np = np.array([np.exp(-0.1 * abs(g)) for g in gammas])

def zeta_noise(t, phase):
    return sum(w * np.cos(g * (t + phase)) for g, w in zip(gammas, weights_np)) / len(gammas)

class ZetaNoiseFilterTask:
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

def train_and_eval(seed, hidden_size=48, epochs=200, zeta_weight=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    vanilla = nn.LSTM(1, hidden_size, batch_first=True)
    zeta = ZetaLSTM(1, hidden_size, M=15, sigma=0.1, zeta_weight=zeta_weight)
    vanilla_out = nn.Linear(hidden_size, 1)
    zeta_out = nn.Linear(hidden_size, 1)

    task = ZetaNoiseFilterTask(100, 0.8)

    v_opt = torch.optim.Adam(list(vanilla.parameters()) + list(vanilla_out.parameters()), lr=2e-3)
    z_opt = torch.optim.Adam(list(zeta.parameters()) + list(zeta_out.parameters()), lr=2e-3)

    for epoch in range(epochs):
        for _ in range(15):
            vanilla.train(); vanilla_out.train()
            x, y = task.generate_batch(32)
            v_opt.zero_grad()
            loss_v = nn.functional.mse_loss(vanilla_out(vanilla(x)[0]), y)
            loss_v.backward()
            v_opt.step()

            zeta.train(); zeta_out.train()
            x, y = task.generate_batch(32)
            z_opt.zero_grad()
            loss_z = nn.functional.mse_loss(zeta_out(zeta(x)[0]), y)
            loss_z.backward()
            z_opt.step()

    # Eval
    vanilla.eval(); zeta.eval()
    v_final, z_final = [], []
    with torch.no_grad():
        for _ in range(50):
            x, y = task.generate_batch(32)
            v_final.append(nn.functional.mse_loss(vanilla_out(vanilla(x)[0]), y).item())
            z_final.append(nn.functional.mse_loss(zeta_out(zeta(x)[0]), y).item())

    return np.mean(v_final), np.mean(z_final)

# Run multiple seeds
seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
print(f'\nRunning {len(seeds)} seeds...')

vanilla_results = []
zeta_results = []

for i, seed in enumerate(seeds):
    v_mse, z_mse = train_and_eval(seed, zeta_weight=0.5)
    vanilla_results.append(v_mse)
    zeta_results.append(z_mse)
    imp = (v_mse - z_mse) / v_mse * 100
    print(f'  Seed {seed}: Vanilla={v_mse:.6f}, Zeta={z_mse:.6f}, Improvement={imp:+.2f}%')

# Aggregate results
v_mean, v_std = np.mean(vanilla_results), np.std(vanilla_results)
z_mean, z_std = np.mean(zeta_results), np.std(zeta_results)
improvements = [(v - z) / v * 100 for v, z in zip(vanilla_results, zeta_results)]
imp_mean, imp_std = np.mean(improvements), np.std(improvements)

# Statistical test
wins = sum(1 for imp in improvements if imp > 0)

print('\n' + '=' * 70)
print('AGGREGATED RESULTS (10 seeds):')
print(f'  Vanilla LSTM:  {v_mean:.6f} (+/- {v_std:.6f})')
print(f'  Zeta LSTM:     {z_mean:.6f} (+/- {z_std:.6f})')
print(f'  Improvement:   {imp_mean:+.2f}% (+/- {imp_std:.2f}%)')
print(f'  Win rate:      {wins}/{len(seeds)} ({wins/len(seeds)*100:.0f}%)')

if imp_mean >= 10:
    print('\n  *** PAPER CONJECTURE (~10%) VALIDATED! ***')
elif imp_mean >= 5:
    print('\n  [OK] Significant average improvement (>5%)!')
elif imp_mean > 0 and wins >= 7:
    print('\n  [OK] Consistent improvement across seeds')
elif imp_mean > 0:
    print('\n  [OK] Average improvement, but inconsistent')
else:
    print('\n  [--] No consistent improvement')
print('=' * 70)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
data = [vanilla_results, zeta_results]
bp = axes[0].boxplot(data, labels=['Vanilla', 'Zeta'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title(f'Distribution over {len(seeds)} seeds')
axes[0].grid(True, alpha=0.3)

# Improvement histogram
axes[1].hist(improvements, bins=10, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].axvline(x=imp_mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {imp_mean:.1f}%')
axes[1].set_xlabel('Improvement (%)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Improvement Distribution (Win rate: {wins}/{len(seeds)})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zeta_lstm_robust.png', dpi=150, bbox_inches='tight')
print('\nSaved: zeta_lstm_robust.png')
