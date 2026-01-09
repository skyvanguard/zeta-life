"""
Experiment: Zeta-Gated Memory (learned gate for zeta contribution)

Hypothesis: Instead of fixed additive integration, learn WHEN to apply
zeta memory via a sigmoid gate. This preserves LSTM dynamics while
allowing selective use of zeta information.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_life.core import get_zeta_zeros

print('=' * 70)
print('ZETA LSTM - Gated Memory Integration')
print('=' * 70)

gammas = get_zeta_zeros(15)

class ZetaGatedMemory(nn.Module):
    """Zeta memory with learned gate for selective application."""
    def __init__(self, hidden_size, M=15, sigma=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        gammas_list = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas_list, dtype=torch.float32))

        # Fixed phi (Abel regularization)
        phi = torch.tensor([np.exp(-sigma * abs(g)) for g in gammas_list], dtype=torch.float32)
        self.register_buffer('phi', phi)

        # Learnable gate: decides how much zeta memory to use
        self.gate_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, t):
        # Compute zeta oscillation
        cos_terms = torch.cos(self.gammas * t)
        zeta_weight = (self.phi * cos_terms).sum() / self.M

        # Raw zeta memory
        m_t = zeta_weight * h

        # Learned gate (per-dimension)
        gate = torch.sigmoid(self.gate_linear(h))

        # Gated output
        return gate * m_t


class ZetaLSTMGated(nn.Module):
    """LSTM with gated zeta memory integration."""
    def __init__(self, input_size, hidden_size, M=15, sigma=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.zeta_memory = ZetaGatedMemory(hidden_size, M=M, sigma=sigma)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h_new, c = self.lstm_cell(x[:, t], (h, c))
            # Gated zeta memory addition
            m_t = self.zeta_memory(h, t)
            h = h_new + m_t  # Gate is inside zeta_memory
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))


# Also test: Per-dimension zeta weights
class ZetaPerDimMemory(nn.Module):
    """Zeta memory with per-dimension learned weights."""
    def __init__(self, hidden_size, M=15, sigma=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        gammas_list = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas_list, dtype=torch.float32))

        # Per-dimension phi weights (initialized from Abel regularization)
        phi_init = np.array([np.exp(-sigma * abs(g)) for g in gammas_list])
        # Expand to [hidden_size, M]
        phi_init = np.tile(phi_init, (hidden_size, 1))
        self.phi = nn.Parameter(torch.tensor(phi_init, dtype=torch.float32))

        self.zeta_weight = 0.4

    def forward(self, h, t):
        # [M]
        cos_terms = torch.cos(self.gammas * t)
        # [hidden_size, M] @ [M] -> [hidden_size]
        weights = (self.phi * cos_terms).sum(dim=1) / self.M
        # [batch, hidden] * [hidden] -> [batch, hidden]
        m_t = h * weights.unsqueeze(0)
        return self.zeta_weight * m_t


class ZetaLSTMPerDim(nn.Module):
    """LSTM with per-dimension zeta weights."""
    def __init__(self, input_size, hidden_size, M=15, sigma=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.zeta_memory = ZetaPerDimMemory(hidden_size, M=M, sigma=sigma)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h_new, c = self.lstm_cell(x[:, t], (h, c))
            m_t = self.zeta_memory(h, t)
            h = h_new + m_t
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))


# Zeta noise task
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


def train_and_eval(model_class, seed, hidden_size=48, epochs=100):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(1, hidden_size, M=15, sigma=0.1)
    out_layer = nn.Linear(hidden_size, 1)

    task = ZetaNoiseFilterTask(100, 0.8)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(out_layer.parameters()), lr=2e-3)

    for epoch in range(epochs):
        for _ in range(10):
            model.train()
            out_layer.train()
            x, y = task.generate_batch(32)
            optimizer.zero_grad()
            loss = F.mse_loss(out_layer(model(x)[0]), y)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    out_layer.eval()
    results = []
    with torch.no_grad():
        for _ in range(30):
            x, y = task.generate_batch(32)
            results.append(F.mse_loss(out_layer(model(x)[0]), y).item())

    return np.mean(results)


# Compare all variants
from zeta_life.core import ZetaLSTM

print('\nComparing architectures across 5 seeds...\n')
seeds = [42, 123, 456, 789, 1011]

results = {
    'Vanilla LSTM': [],
    'Fixed Zeta': [],
    'Gated Zeta': [],
    'PerDim Zeta': []
}

for seed in seeds:
    print(f'Seed {seed}:')

    # Vanilla
    torch.manual_seed(seed)
    np.random.seed(seed)
    vanilla = nn.LSTM(1, 48, batch_first=True)
    v_out = nn.Linear(48, 1)
    task = ZetaNoiseFilterTask(100, 0.8)
    v_opt = torch.optim.Adam(list(vanilla.parameters()) + list(v_out.parameters()), lr=2e-3)
    for _ in range(100):
        for _ in range(10):
            x, y = task.generate_batch(32)
            v_opt.zero_grad()
            loss = F.mse_loss(v_out(vanilla(x)[0]), y)
            loss.backward()
            v_opt.step()
    vanilla.eval()
    v_final = []
    with torch.no_grad():
        for _ in range(30):
            x, y = task.generate_batch(32)
            v_final.append(F.mse_loss(v_out(vanilla(x)[0]), y).item())
    v_mse = np.mean(v_final)
    results['Vanilla LSTM'].append(v_mse)

    # Fixed Zeta
    f_mse = train_and_eval(ZetaLSTM, seed)
    results['Fixed Zeta'].append(f_mse)

    # Gated Zeta
    g_mse = train_and_eval(ZetaLSTMGated, seed)
    results['Gated Zeta'].append(g_mse)

    # PerDim Zeta
    p_mse = train_and_eval(ZetaLSTMPerDim, seed)
    results['PerDim Zeta'].append(p_mse)

    print(f'  Vanilla: {v_mse:.6f}')
    print(f'  Fixed:   {f_mse:.6f} ({(v_mse-f_mse)/v_mse*100:+.1f}%)')
    print(f'  Gated:   {g_mse:.6f} ({(v_mse-g_mse)/v_mse*100:+.1f}%)')
    print(f'  PerDim:  {p_mse:.6f} ({(v_mse-p_mse)/v_mse*100:+.1f}%)')
    print()

# Aggregate
print('=' * 70)
print('AGGREGATE RESULTS:')
print('-' * 70)

for name, vals in results.items():
    mean = np.mean(vals)
    std = np.std(vals)
    if name == 'Vanilla LSTM':
        print(f'{name:15s}: {mean:.6f} (+/- {std:.6f})')
        vanilla_mean = mean
    else:
        imp = (vanilla_mean - mean) / vanilla_mean * 100
        wins = sum(1 for v, z in zip(results['Vanilla LSTM'], vals) if z < v)
        print(f'{name:15s}: {mean:.6f} (+/- {std:.6f})  [{imp:+.2f}% vs vanilla, {wins}/5 wins]')

print('=' * 70)

# Find best
best_name = min(['Fixed Zeta', 'Gated Zeta', 'PerDim Zeta'],
                key=lambda n: np.mean(results[n]))
best_imp = (vanilla_mean - np.mean(results[best_name])) / vanilla_mean * 100
print(f'\nBest variant: {best_name} ({best_imp:+.2f}% vs vanilla)')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
data = [results['Vanilla LSTM'], results['Fixed Zeta'],
        results['Gated Zeta'], results['PerDim Zeta']]
labels = ['Vanilla', 'Fixed', 'Gated', 'PerDim']
bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'salmon', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Architecture Comparison (5 seeds)')
axes[0].grid(True, alpha=0.3)

# Improvement per variant
improvements = {}
for name in ['Fixed Zeta', 'Gated Zeta', 'PerDim Zeta']:
    improvements[name] = [(v - z) / v * 100
                          for v, z in zip(results['Vanilla LSTM'], results[name])]

x = np.arange(len(seeds))
width = 0.25
axes[1].bar(x - width, improvements['Fixed Zeta'], width, label='Fixed', alpha=0.7)
axes[1].bar(x, improvements['Gated Zeta'], width, label='Gated', alpha=0.7)
axes[1].bar(x + width, improvements['PerDim Zeta'], width, label='PerDim', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Seed')
axes[1].set_ylabel('Improvement vs Vanilla (%)')
axes[1].set_title('Per-Seed Improvement by Variant')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(s) for s in seeds])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zeta_lstm_gated_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved: zeta_lstm_gated_comparison.png')
