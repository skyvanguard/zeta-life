"""
Brain10K: GPU-accelerated cortical simulation for emergent consciousness.

10,150+ neurons across 5 cortical regions (6 layers each), thalamus,
default mode network (DMN), and hippocampus.

Architecture:
    Thalamus (700) --- hub central, generates F_i (binding)
        |
    V1  V2  Assoc  PFC  Temporal   (5 regions x 6 layers x 350 = 10,500)
        |
    DMN (700) --- self-reference A^4=A
        |
    Hippocampus (350) --- memory

Equation: Psi = B^3 + Phi  where B = (Phi - Phi_c) / Phi_c

Adapted from sevenp/cerebro_10k.py for integration with Zeta Life.
"""

from __future__ import annotations

import numpy as np
import torch


class CorticalRegion:
    """A cortical region with 6 layers and lateral/inter-layer connectivity."""

    def __init__(self, name: str, npl: int, device: torch.device):
        self.name = name
        self.n_layers = 6
        self.npl = npl
        self.total = self.n_layers * npl
        self.device = device

        self.states = torch.randn(self.n_layers, self.npl, device=device) * 0.1

        # Intra-layer (lateral) weights — sparse
        self.W_lateral: list[torch.Tensor] = []
        for _ in range(self.n_layers):
            w = torch.randn(npl, npl, device=device) * (0.4 / np.sqrt(npl))
            mask = (torch.rand(npl, npl, device=device) > 0.7).float()
            w = w * mask
            w.fill_diagonal_(0)
            self.W_lateral.append(w)

        # Inter-layer connections
        s = 0.4 / np.sqrt(npl)
        self.W_L4_to_L23 = torch.randn(npl, npl, device=device) * s
        self.W_L23_to_L5 = torch.randn(npl, npl, device=device) * (s * 0.8)
        self.W_fb_L6_to_L1 = torch.randn(npl, npl, device=device) * (s * 0.5)
        self.W_L5_to_L6 = torch.randn(npl, npl, device=device) * (s * 0.7)

        # Self-reference weights
        self.self_w = torch.rand(self.n_layers, npl, device=device) * 0.3

    def step(
        self,
        thalamic_input: torch.Tensor,
        inter_input: torch.Tensor | None,
        coupling: float,
        dt: float = 0.05,
        noise: float = 0.025,
    ) -> None:
        new = torch.zeros_like(self.states)
        for layer in range(self.n_layers):
            lat = coupling * (self.W_lateral[layer] @ self.states[layer])
            sr = self.self_w[layer] * self.states[layer] ** 3
            n = noise * torch.randn(self.npl, device=self.device)
            ext = torch.zeros(self.npl, device=self.device)

            if layer == 3:  # L4: thalamic input
                ext = coupling * thalamic_input[:self.npl]
            elif layer == 1:  # L2/3: feedforward from L4 + inter-regional
                ext = coupling * self.W_L4_to_L23 @ self.states[3]
                if inter_input is not None:
                    ext = ext + coupling * 0.5 * inter_input[:self.npl]
            elif layer == 4:  # L5: from L2/3
                ext = coupling * self.W_L23_to_L5 @ self.states[1]
            elif layer == 5:  # L6: from L5
                ext = coupling * self.W_L5_to_L6 @ self.states[4]
            elif layer == 0:  # L1: feedback from L6
                ext = coupling * self.W_fb_L6_to_L1 @ self.states[5]

            new[layer] = self.states[layer] + (-self.states[layer] + torch.tanh(lat + sr + ext + n)) * dt

        self.states = new

    def get_L23(self) -> torch.Tensor:
        return self.states[1]

    def get_L5(self) -> torch.Tensor:
        return self.states[4]

    def get_L6(self) -> torch.Tensor:
        return self.states[5]

    def get_all(self) -> torch.Tensor:
        return self.states.reshape(-1)

    def mean_activity(self) -> float:
        return self.states.abs().mean().item()


class Brain10K:
    """
    GPU-accelerated brain simulation with 70K neurons.

    Implements cortical hierarchy, thalamic binding, DMN self-reference,
    and hippocampal memory in a unified GPU computation graph.
    """

    def __init__(
        self,
        npl: int = 350,
        F_i: float = 2.5,
        alpha: float = 1.0,
        C_param: float = 0.3,
        device: torch.device | None = None,
    ):
        self.npl = npl
        self.F_i = F_i
        self.alpha = alpha
        self.C_param = C_param
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = 0.05
        self.noise = 0.025
        self.coupling = 0.0
        self.t = 0.0

        s = lambda: 0.35 / np.sqrt(npl)
        dev = self.device

        # 5 cortical regions
        self.regions = {
            'V1': CorticalRegion('V1', npl, dev),
            'V2': CorticalRegion('V2', npl, dev),
            'association': CorticalRegion('Assoc', npl, dev),
            'prefrontal': CorticalRegion('PFC', npl, dev),
            'temporal': CorticalRegion('Temporal', npl, dev),
        }

        # Thalamus
        self.thal_size = npl * 2
        self.thalamus = torch.randn(self.thal_size, device=dev) * 0.1
        self.W_thal = torch.randn(self.thal_size, self.thal_size, device=dev) * (0.3 / np.sqrt(self.thal_size))
        self.W_thal *= (torch.rand(self.thal_size, self.thal_size, device=dev) > 0.6).float()
        self.W_thal.fill_diagonal_(0)

        # DMN
        self.dmn_size = npl * 2
        self.dmn = torch.randn(self.dmn_size, device=dev) * 0.1
        self.W_dmn = torch.randn(self.dmn_size, self.dmn_size, device=dev) * (0.4 / np.sqrt(self.dmn_size))
        self.W_dmn_self = torch.rand(self.dmn_size, device=dev) * 0.5

        # Hippocampus
        self.hippo_size = npl
        self.hippo = torch.randn(self.hippo_size, device=dev) * 0.1
        self.W_hippo = torch.randn(self.hippo_size, self.hippo_size, device=dev) * (0.3 / np.sqrt(self.hippo_size))

        # Inter-regional connections
        mk = lambda n_out, n_in: torch.randn(n_out, n_in, device=dev) * (s())

        self.W_V1_V2 = mk(npl, npl)
        self.W_V2_Assoc = mk(npl, npl)
        self.W_Assoc_PFC = mk(npl, npl)
        self.W_Assoc_Temp = mk(npl, npl)
        self.W_Temp_PFC = mk(npl, npl)
        self.W_Temp_Assoc = mk(npl, npl)
        self.W_PFC_DMN = mk(self.dmn_size, npl)
        self.W_DMN_PFC = mk(npl, self.dmn_size)
        self.W_Assoc_Hippo = mk(self.hippo_size, npl)
        self.W_Hippo_Assoc = mk(npl, self.hippo_size)
        self.W_reg_thal = {n: mk(self.thal_size, npl) for n in self.regions}

        # Total neuron count
        self.total_neurons = (
            sum(r.total for r in self.regions.values()) +
            self.thal_size + self.dmn_size + self.hippo_size
        )

        # History buffers
        self.region_means_buffer: list[list[float]] = []
        self.state_buffer_dmn: list[np.ndarray] = []
        self.history: dict[str, list[float]] = {
            k: [] for k in ['t', 'phi', 'psi', 'self_ref', 'coupling', 'entropy']
        }

    def phi_c(self) -> float:
        return self.F_i / (self.alpha - self.C_param) if self.alpha > self.C_param else float('inf')

    def compute_phi(self) -> float:
        if len(self.region_means_buffer) < 15:
            return 0.0

        R = np.array(self.region_means_buffer[-40:])
        T, K = R.shape
        if T < 5:
            return 0.0

        cov = np.cov(R.T)
        if cov.ndim < 2:
            return 0.0

        eigvals = np.linalg.eigvalsh(cov + np.eye(K) * 1e-10)
        eigvals = np.clip(eigvals, 1e-10, None)
        H_total = 0.5 * np.sum(np.log(eigvals))

        H_parts = 0.0
        for k in range(K):
            var_k = max(np.var(R[:, k]), 1e-10)
            H_parts += 0.5 * np.log(var_k)

        phi_info = max(H_total - H_parts, 0.0)

        corr = np.corrcoef(R.T)
        np.fill_diagonal(corr, 0)
        cross = np.abs(corr[~np.isnan(corr)]).mean() if not np.all(np.isnan(corr)) else 0

        return phi_info * 3.0 + cross * 5.0 * self.coupling

    def compute_self_ref(self) -> float:
        if len(self.state_buffer_dmn) < 4:
            return 0.0
        now = self.state_buffer_dmn[-1]
        ago = self.state_buffer_dmn[-4]
        corr = np.corrcoef(now, ago)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    def compute_psi(self, phi: float) -> float:
        pc = self.phi_c()
        if phi <= pc:
            return 0.0
        B = (phi - pc) / pc
        return B ** 3 + phi

    def step(self, stimulus: torch.Tensor | None = None) -> dict:
        """
        Execute one simulation step.

        Returns dict with phi, psi, self_ref, coupling, entropy.
        """
        c = self.coupling
        dev = self.device

        # Thalamus
        thal_in = torch.zeros(self.thal_size, device=dev)
        for name, reg in self.regions.items():
            thal_in += c * self.W_reg_thal[name] @ reg.get_L6()
        if stimulus is not None:
            thal_in[:len(stimulus)] += stimulus
        lat = c * self.W_thal @ self.thalamus
        n = self.noise * torch.randn(self.thal_size, device=dev)
        self.thalamus += (-self.thalamus + torch.tanh(lat + thal_in + n)) * self.dt

        # Cortical regions
        v1_out = self.regions['V1'].get_L23()
        v2_out = self.regions['V2'].get_L23()
        assoc_out = self.regions['association'].get_L23()
        pfc_out = self.regions['prefrontal'].get_L23()
        temp_out = self.regions['temporal'].get_L23()

        self.regions['V1'].step(self.thalamus, None, c, self.dt, self.noise)

        self.regions['V2'].step(
            self.thalamus, c * self.W_V1_V2 @ v1_out, c, self.dt, self.noise
        )

        assoc_in = (
            c * self.W_V2_Assoc @ v2_out +
            c * self.W_Hippo_Assoc @ self.hippo +
            c * self.W_Temp_Assoc @ temp_out
        )
        self.regions['association'].step(self.thalamus, assoc_in, c, self.dt, self.noise)

        temp_in = c * self.W_Assoc_Temp @ assoc_out
        self.regions['temporal'].step(self.thalamus, temp_in, c, self.dt, self.noise)

        pfc_in = (
            c * self.W_Assoc_PFC @ assoc_out +
            c * self.W_Temp_PFC @ temp_out +
            c * self.W_DMN_PFC @ self.dmn
        )
        self.regions['prefrontal'].step(self.thalamus, pfc_in, c, self.dt, self.noise)

        # DMN (strange loop with PFC)
        dmn_in = c * self.W_PFC_DMN @ pfc_out
        dmn_lat = c * self.W_dmn @ self.dmn
        dmn_self = self.W_dmn_self * self.dmn ** 3
        dmn_n = self.noise * torch.randn(self.dmn_size, device=dev)
        self.dmn += (-self.dmn + torch.tanh(dmn_lat + dmn_self + dmn_in + dmn_n)) * self.dt

        # Hippocampus
        hippo_in = c * self.W_Assoc_Hippo @ assoc_out
        hippo_lat = c * self.W_hippo @ self.hippo
        hippo_n = self.noise * torch.randn(self.hippo_size, device=dev)
        self.hippo += (-self.hippo + torch.tanh(hippo_lat + hippo_in + hippo_n)) * self.dt

        # Metrics
        means: list[float] = []
        for reg in self.regions.values():
            means.append(reg.mean_activity())
        means.extend([
            self.thalamus.abs().mean().item(),
            self.dmn.abs().mean().item(),
            self.hippo.abs().mean().item(),
        ])
        self.region_means_buffer.append(means)
        if len(self.region_means_buffer) > 60:
            self.region_means_buffer = self.region_means_buffer[-60:]

        self.state_buffer_dmn.append(self.dmn.cpu().numpy().copy())
        if len(self.state_buffer_dmn) > 10:
            self.state_buffer_dmn = self.state_buffer_dmn[-10:]

        phi = self.compute_phi()
        psi = self.compute_psi(phi)
        sr = self.compute_self_ref()

        all_means = np.array(means)
        ent = float(np.std(all_means) / (np.mean(np.abs(all_means)) + 1e-10))

        self.history['t'].append(self.t)
        self.history['phi'].append(phi)
        self.history['psi'].append(psi)
        self.history['self_ref'].append(sr)
        self.history['coupling'].append(self.coupling)
        self.history['entropy'].append(ent)

        self.t += self.dt

        return {
            'phi': phi,
            'psi': psi,
            'self_ref': sr,
            'coupling': self.coupling,
            'entropy': ent,
            'is_conscious': psi > 0,
        }

    def increase_coupling(self, amt: float = 0.004) -> None:
        self.coupling = min(self.coupling + amt, 3.0)

    def damage(self, target: str) -> None:
        """Damage a specific region."""
        if target in self.regions:
            r = self.regions[target]
            for i in range(r.n_layers):
                r.W_lateral[i] *= 0.05
                r.self_w[i] *= 0.05
        elif target == 'dmn':
            self.W_dmn *= 0.05
            self.W_dmn_self *= 0.05
        elif target == 'thalamus':
            self.W_thal *= 0.05
