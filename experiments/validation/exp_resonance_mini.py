"""Experimento minimo - 1 seed, 30 epochs"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_life.core import ZetaLSTM, get_zeta_zeros
from zeta_life.core import ZetaLSTMResonantSimple

print('Test rapido ZetaLSTM Resonante...')

gammas = get_zeta_zeros(15)
weights_np = np.array([np.exp(-0.1 * abs(g)) for g in gammas])

def zeta_noise(t, phase):
    return sum(w * np.cos(g * (t + phase)) for g, w in zip(gammas, weights_np)) / len(gammas)

# Generar datos
def gen_batch(batch_size=32, seq_len=50):
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

seed = 42
hidden = 32
epochs = 30

# Vanilla
torch.manual_seed(seed); np.random.seed(seed)
v_model = nn.LSTM(1, hidden, batch_first=True)
v_out = nn.Linear(hidden, 1)
v_opt = torch.optim.Adam(list(v_model.parameters()) + list(v_out.parameters()), lr=2e-3)

for _ in range(epochs):
    for _ in range(5):
        x, y = gen_batch()
        v_opt.zero_grad()
        loss = F.mse_loss(v_out(v_model(x)[0]), y)
        loss.backward()
        v_opt.step()

v_model.eval()
v_loss = []
with torch.no_grad():
    for _ in range(10):
        x, y = gen_batch()
        v_loss.append(F.mse_loss(v_out(v_model(x)[0]), y).item())
v_mean = np.mean(v_loss)
print(f'Vanilla: {v_mean:.6f}')

# Original
torch.manual_seed(seed); np.random.seed(seed)
o_model = ZetaLSTM(1, hidden, M=15, sigma=0.1, zeta_weight=0.4)
o_out = nn.Linear(hidden, 1)
o_opt = torch.optim.Adam(list(o_model.parameters()) + list(o_out.parameters()), lr=2e-3)

for _ in range(epochs):
    for _ in range(5):
        x, y = gen_batch()
        o_opt.zero_grad()
        loss = F.mse_loss(o_out(o_model(x)[0]), y)
        loss.backward()
        o_opt.step()

o_model.eval()
o_loss = []
with torch.no_grad():
    for _ in range(10):
        x, y = gen_batch()
        o_loss.append(F.mse_loss(o_out(o_model(x)[0]), y).item())
o_mean = np.mean(o_loss)
imp_o = (v_mean - o_mean) / v_mean * 100
print(f'Original: {o_mean:.6f} ({imp_o:+.1f}% vs vanilla)')

# Resonante
torch.manual_seed(seed); np.random.seed(seed)
r_model = ZetaLSTMResonantSimple(1, hidden, M=15, sigma=0.1)
r_out = nn.Linear(hidden, 1)
r_opt = torch.optim.Adam(list(r_model.parameters()) + list(r_out.parameters()), lr=2e-3)

for _ in range(epochs):
    for _ in range(5):
        x, y = gen_batch()
        r_opt.zero_grad()
        loss = F.mse_loss(r_out(r_model(x)[0]), y)
        loss.backward()
        r_opt.step()

r_model.eval()
r_loss = []
with torch.no_grad():
    for _ in range(10):
        x, y = gen_batch()
        r_loss.append(F.mse_loss(r_out(r_model(x)[0]), y).item())
r_mean = np.mean(r_loss)
imp_r = (v_mean - r_mean) / v_mean * 100
imp_r_vs_o = (o_mean - r_mean) / o_mean * 100
print(f'Resonant: {r_mean:.6f} ({imp_r:+.1f}% vs vanilla, {imp_r_vs_o:+.1f}% vs original)')

print('\nDone!')
