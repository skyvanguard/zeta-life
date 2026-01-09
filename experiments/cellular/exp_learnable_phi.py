"""
Experiment: Constrained Learnable Phi for ZetaLSTM
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
print('ZETA LSTM - Constrained Learnable Phi')
print('=' * 70)

gammas = get_zeta_zeros(15)

class ZetaMemoryLayerConstrained(nn.Module):
    """Zeta memory with positive-constrained learnable phi using softplus."""
    def __init__(self, hidden_size, M=15, sigma=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        gammas_list = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas_list, dtype=torch.float32))

        # Initialize log_phi so softplus(log_phi) ~ exp(-sigma*|gamma|)
        phi_init = np.array([np.exp(-sigma * abs(g)) for g in gammas_list])
        log_phi_init = np.log(np.exp(phi_init) - 1 + 1e-6)
        self.log_phi = nn.Parameter(torch.tensor(log_phi_init, dtype=torch.float32))

    def forward(self, h, t):
        phi = F.softplus(self.log_phi)
        cos_terms = torch.cos(self.gammas * t)
        weights = (phi * cos_terms).sum() / self.M
        return weights * h

    @property
    def phi(self):
        return F.softplus(self.log_phi)

class ZetaLSTMConstrained(nn.Module):
    def __init__(self, input_size, hidden_size, M=15, sigma=0.1, zeta_weight=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.zeta_memory = ZetaMemoryLayerConstrained(hidden_size, M=M, sigma=sigma)
        self.zeta_weight = zeta_weight

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h_new, c = self.lstm_cell(x[:, t], (h, c))
            m_t = self.zeta_memory(h, t)
            h = h_new + self.zeta_weight * m_t
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

# Config
hidden_size = 48
epochs = 300
M = 15

print(f'Config: hidden={hidden_size}, epochs={epochs}, M={M}')
print('Phi constrained to positive values via softplus')

# Models
vanilla = nn.LSTM(1, hidden_size, batch_first=True)
zeta_fixed = ZetaLSTM(1, hidden_size, M=M, sigma=0.1, zeta_weight=0.4)
zeta_const = ZetaLSTMConstrained(1, hidden_size, M=M, sigma=0.1, zeta_weight=0.4)

vanilla_out = nn.Linear(hidden_size, 1)
fixed_out = nn.Linear(hidden_size, 1)
const_out = nn.Linear(hidden_size, 1)

task = ZetaNoiseFilterTask(100, 0.8)

# Separate learning rates
lstm_params = list(zeta_const.lstm_cell.parameters())
phi_params = [zeta_const.zeta_memory.log_phi]

v_opt = torch.optim.Adam(list(vanilla.parameters()) + list(vanilla_out.parameters()), lr=2e-3)
f_opt = torch.optim.Adam(list(zeta_fixed.parameters()) + list(fixed_out.parameters()), lr=2e-3)
c_opt = torch.optim.Adam([
    {'params': lstm_params + list(const_out.parameters()), 'lr': 2e-3},
    {'params': phi_params, 'lr': 5e-4}
])

v_losses, f_losses, c_losses = [], [], []

print('\nTraining...')
for epoch in range(epochs):
    for _ in range(15):
        vanilla.train(); vanilla_out.train()
        x, y = task.generate_batch(32)
        v_opt.zero_grad()
        loss_v = nn.functional.mse_loss(vanilla_out(vanilla(x)[0]), y)
        loss_v.backward()
        v_opt.step()

        zeta_fixed.train(); fixed_out.train()
        x, y = task.generate_batch(32)
        f_opt.zero_grad()
        loss_f = nn.functional.mse_loss(fixed_out(zeta_fixed(x)[0]), y)
        loss_f.backward()
        f_opt.step()

        zeta_const.train(); const_out.train()
        x, y = task.generate_batch(32)
        c_opt.zero_grad()
        loss_c = nn.functional.mse_loss(const_out(zeta_const(x)[0]), y)
        loss_c.backward()
        c_opt.step()

    v_losses.append(loss_v.item())
    f_losses.append(loss_f.item())
    c_losses.append(loss_c.item())

    if (epoch + 1) % 60 == 0:
        print(f'  Epoch {epoch+1}: Vanilla={loss_v.item():.4f}, Fixed={loss_f.item():.4f}, Constrained={loss_c.item():.4f}')

# Evaluation
vanilla.eval(); zeta_fixed.eval(); zeta_const.eval()
v_final, f_final, c_final = [], [], []

with torch.no_grad():
    for _ in range(100):
        x, y = task.generate_batch(32)
        v_final.append(nn.functional.mse_loss(vanilla_out(vanilla(x)[0]), y).item())
        f_final.append(nn.functional.mse_loss(fixed_out(zeta_fixed(x)[0]), y).item())
        c_final.append(nn.functional.mse_loss(const_out(zeta_const(x)[0]), y).item())

v_avg, f_avg, c_avg = np.mean(v_final), np.mean(f_final), np.mean(c_final)
v_std, f_std, c_std = np.std(v_final), np.std(f_final), np.std(c_final)

imp_fixed = (v_avg - f_avg) / v_avg * 100
imp_const = (v_avg - c_avg) / v_avg * 100

print('\n' + '=' * 70)
print('RESULTS:')
print(f'  Vanilla LSTM:        {v_avg:.6f} (+/- {v_std:.6f})')
print(f'  Zeta Fixed Phi:      {f_avg:.6f} (+/- {f_std:.6f})  [{imp_fixed:+.2f}% vs vanilla]')
print(f'  Zeta Constrained:    {c_avg:.6f} (+/- {c_std:.6f})  [{imp_const:+.2f}% vs vanilla]')

if imp_const >= 10:
    print('\n  *** PAPER CONJECTURE (~10%) VALIDATED! ***')
elif imp_const > imp_fixed:
    print('\n  [OK] Constrained learnable phi improves over fixed!')
elif imp_const > 0:
    print('\n  [OK] Still better than vanilla')
print('=' * 70)

# Show learned phi
print('\nPhi weights (constrained positive):')
initial_phi = np.array([np.exp(-0.1 * abs(g)) for g in gammas])
learned_phi = zeta_const.zeta_memory.phi.detach().numpy()
for i in range(min(8, len(gammas))):
    ratio = learned_phi[i] / initial_phi[i]
    print(f'  gamma_{i+1}={gammas[i]:.1f}: init={initial_phi[i]:.4f}, learned={learned_phi[i]:.4f}, ratio={ratio:.2f}x')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(v_losses, label='Vanilla', alpha=0.7)
axes[0].plot(f_losses, label='Fixed Phi', alpha=0.7)
axes[0].plot(c_losses, label='Constrained Learn', alpha=0.7)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

models = ['Vanilla', 'Fixed', 'Constrained']
means = [v_avg, f_avg, c_avg]
stds = [v_std, f_std, c_std]
colors = ['blue', 'green', 'red']
axes[1].bar(models, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Final Performance')
axes[1].grid(True, alpha=0.3, axis='y')

x_pos = np.arange(len(gammas))
width = 0.35
axes[2].bar(x_pos - width/2, initial_phi, width, label='Initial', alpha=0.7)
axes[2].bar(x_pos + width/2, learned_phi, width, label='Learned', alpha=0.7)
axes[2].set_xlabel('Zeta Zero Index')
axes[2].set_ylabel('Phi Weight')
axes[2].set_title('Constrained Phi: Initial vs Learned')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zeta_lstm_constrained_phi.png', dpi=150, bbox_inches='tight')
print('\nSaved: zeta_lstm_constrained_phi.png')
