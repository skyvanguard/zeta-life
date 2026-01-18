"""
Zeta Resonance Detector - Nueva arquitectura basada en notas de investigacion

Principio: En lugar de IMPONER modulacion zeta, DETECTAR cuando los datos
resuenan con frecuencias zeta y aplicar memoria solo entonces.

Inspirado por:
- "Marcadores de tension" en distribucion de primos
- "Conciencia como conjunto de restricciones"
- "El orden oculto solo existe si los ceros estan en linea critica"
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_zeta_zeros(M: int = 15) -> list[float]:
    """Primeros M ceros no triviales de zeta (parte imaginaria)."""
    zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
        67.079811, 69.546402, 72.067158, 75.704691, 77.144840
    ]
    return zeros[:M]


class ZetaSpectrumAnalyzer(nn.Module):
    """
    Analiza el espectro de la entrada y detecta resonancia con frecuencias zeta.

    En lugar de calcular FFT completa, usa correlacion directa con
    las frecuencias zeta conocidas - mas eficiente y directo.
    """
    def __init__(self, input_size: int, M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.M = M

        # Frecuencias zeta
        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))

        # Pesos de regularizacion Abel
        phi = torch.tensor([np.exp(-sigma * abs(g)) for g in gammas], dtype=torch.float32)
        self.register_buffer('phi', phi)

        # Proyeccion aprendible de entrada a espacio de analisis
        self.input_proj = nn.Linear(input_size, M * 2)  # cos y sin para cada gamma

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Calcula resonancia de x con frecuencias zeta en tiempo t.

        Args:
            x: [batch, input_size] - entrada actual
            t: timestep actual

        Returns:
            resonance: [batch, 1] - nivel de resonancia (0-1)
        """
        batch_size = x.shape[0]

        # Proyectar entrada a espacio de frecuencias
        proj = self.input_proj(x)  # [batch, M*2]
        cos_coeffs = proj[:, :self.M]  # [batch, M]
        sin_coeffs = proj[:, self.M:]  # [batch, M]

        # Calcular correlacion con osciladores zeta
        # Osciladores teoricos en tiempo t
        zeta_cos = torch.cos(self.gammas * t)  # type: ignore[operator, arg-type]  # [M]
        zeta_sin = torch.sin(self.gammas * t)  # type: ignore[operator, arg-type]  # [M]

        # Correlacion ponderada por phi (Abel regularization)
        cos_corr = (cos_coeffs * zeta_cos * self.phi).sum(dim=1)  # [batch]
        sin_corr = (sin_coeffs * zeta_sin * self.phi).sum(dim=1)  # [batch]

        # Magnitud de resonancia
        resonance = torch.sqrt(cos_corr**2 + sin_corr**2 + 1e-8)  # [batch]

        # Normalizar a [0, 1]
        resonance = torch.sigmoid(resonance - 0.5)  # threshold adaptativo

        return resonance.unsqueeze(1)  # [batch, 1]


class TensionMarkerDetector(nn.Module):
    """
    Detecta "marcadores de tension" - transiciones donde el patron cambia.

    Basado en la observacion de que en la distribucion de primos,
    hay puntos donde el patron "normal" se rompe.
    """
    def __init__(self, hidden_size: int, window_size: int = 5) -> None:
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size

        # Red para detectar cambios en patrones
        self.change_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Buffer para historial reciente
        self.register_buffer('history', None)

    def reset_history(self, batch_size: int, device: torch.device) -> None:
        """Reinicia el historial para nuevo batch."""
        self.history = torch.zeros(batch_size, self.window_size, self.hidden_size, device=device)

    def forward(self, h: torch.Tensor, t: int) -> torch.Tensor:
        """
        Detecta si el estado actual h representa un "marcador de tension".

        Args:
            h: [batch, hidden] - estado oculto actual
            t: timestep

        Returns:
            tension: [batch, 1] - nivel de tension (0-1)
        """
        batch_size = h.shape[0]

        if self.history is None or self.history.shape[0] != batch_size:
            self.reset_history(batch_size, h.device)

        # Actualizar historial (circular buffer)
        idx = t % self.window_size
        self.history[:, idx] = h.detach()

        if t < 2:
            return torch.zeros(batch_size, 1, device=h.device)

        # Calcular promedio historico
        h_mean = self.history.mean(dim=1)  # [batch, hidden]

        # Comparar actual con promedio
        comparison = torch.cat([h, h_mean], dim=1)  # [batch, hidden*2]

        # Detectar desviacion (tension)
        tension: torch.Tensor = self.change_detector(comparison)

        return tension


class ZetaMemoryGated(nn.Module):
    """
    Memoria zeta con gate basado en resonancia y tension.

    Solo aplica memoria zeta cuando:
    1. Hay resonancia con frecuencias zeta
    2. O hay un "marcador de tension"
    """
    def __init__(self, hidden_size: int, M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))

        phi = torch.tensor([np.exp(-sigma * abs(g)) for g in gammas], dtype=torch.float32)
        self.register_buffer('phi', phi)

        # Gate combinator: combina resonancia y tension
        self.gate_combiner = nn.Linear(2, 1)

    def forward(self, h: torch.Tensor, t: int,
                resonance: torch.Tensor, tension: torch.Tensor) -> torch.Tensor:
        """
        Calcula memoria zeta gateada.

        Args:
            h: [batch, hidden] - estado oculto
            t: timestep
            resonance: [batch, 1] - nivel de resonancia
            tension: [batch, 1] - nivel de tension

        Returns:
            m_t: [batch, hidden] - memoria zeta gateada
        """
        # Calcular oscilacion zeta base
        cos_terms = torch.cos(self.gammas * t)  # type: ignore[operator, arg-type]
        zeta_weight = (self.phi * cos_terms).sum() / self.M  # type: ignore[operator]

        # Memoria zeta raw
        m_raw = zeta_weight * h

        # Combinar resonancia y tension para gate final
        gate_input = torch.cat([resonance, tension], dim=1)  # [batch, 2]
        gate = torch.sigmoid(self.gate_combiner(gate_input))  # [batch, 1]

        # Aplicar gate
        m_t = gate * m_raw

        return m_t


class ZetaMemoryGatedSimple(nn.Module):
    """
    Memoria zeta con gate aprendido - version simplificada para OrganismCell.

    Solo requiere la percepcion como entrada, internamente mantiene
    el tiempo y calcula el gate.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Ceros zeta
        gammas = get_zeta_zeros(M)
        weights = [np.exp(-sigma * abs(g)) for g in gammas]
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))
        self.register_buffer('phi', torch.tensor(weights, dtype=torch.float32))

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Memory transformation
        self.memory_net = nn.Linear(input_dim, input_dim)

        # Internal time counter
        self.t = 0

    def reset_time(self) -> None:
        """Reset internal time counter."""
        self.t = 0

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Calcula memoria zeta gateada.

        Args:
            x: [batch, input_dim] - percepcion

        Returns:
            memory: [batch, input_dim] - memoria gateada
            gate: float - valor del gate (promedio del batch)
        """
        self.t += 1

        # Oscilacion zeta en tiempo t
        oscillation = (self.phi * torch.cos(self.gammas * self.t)).sum()  # type: ignore[operator, arg-type]

        # Transformar entrada con modulacion zeta
        zeta_mod = self.memory_net(x) * oscillation

        # Gate aprendido
        gate = self.gate_net(x)

        # Memoria gateada
        memory = gate * zeta_mod

        return memory, gate


class ZetaLSTMResonant(nn.Module):
    """
    LSTM con memoria zeta basada en resonancia.

    Arquitectura:
    1. LSTM cell procesa entrada
    2. ZetaSpectrumAnalyzer detecta resonancia
    3. TensionMarkerDetector detecta transiciones
    4. ZetaMemoryGated aplica memoria solo cuando es relevante

    Principio: "No imponer, detectar"
    """
    def __init__(self, input_size: int, hidden_size: int,
                 M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        # Componentes
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.spectrum_analyzer = ZetaSpectrumAnalyzer(input_size, M, sigma)
        self.tension_detector = TensionMarkerDetector(hidden_size)
        self.zeta_memory = ZetaMemoryGated(hidden_size, M, sigma)

        # Diagnosticos
        self.last_resonances: list[float] = []
        self.last_tensions: list[float] = []
        self.last_gates: list[float] = []

    def forward(self, x: torch.Tensor,
                return_diagnostics: bool = False) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_size]
            return_diagnostics: si True, retorna info de gates

        Returns:
            output: [batch, seq_len, hidden_size]
            (h_n, c_n): estados finales
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        # Reset tension detector
        self.tension_detector.reset_history(batch_size, device)

        outputs = []
        self.last_resonances = []
        self.last_tensions = []

        for t in range(seq_len):
            x_t = x[:, t]  # [batch, input_size]

            # 1. LSTM step
            h_new, c = self.lstm_cell(x_t, (h, c))

            # 2. Detectar resonancia con zeta
            resonance = self.spectrum_analyzer(x_t, t)  # [batch, 1]

            # 3. Detectar marcador de tension
            tension = self.tension_detector(h, t)  # [batch, 1]

            # 4. Memoria zeta gateada
            m_t = self.zeta_memory(h, t, resonance, tension)  # [batch, hidden]

            # 5. Combinar
            h = h_new + m_t

            outputs.append(h)

            if return_diagnostics:
                self.last_resonances.append(resonance.mean().item())
                self.last_tensions.append(tension.mean().item())

        output = torch.stack(outputs, dim=1)
        return output, (h.unsqueeze(0), c.unsqueeze(0))

    def get_diagnostics(self) -> dict:
        """Retorna diagnosticos del ultimo forward."""
        return {
            'resonances': self.last_resonances,
            'tensions': self.last_tensions,
            'avg_resonance': np.mean(self.last_resonances) if self.last_resonances else 0,
            'avg_tension': np.mean(self.last_tensions) if self.last_tensions else 0
        }


# Version simplificada para comparacion rapida
class ZetaLSTMResonantSimple(nn.Module):
    """
    Version simplificada del ZetaLSTM resonante.

    Solo usa deteccion de resonancia espectral, sin detector de tension.
    Mas rapido de entrenar para experimentos.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M

        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))

        phi = torch.tensor([np.exp(-sigma * abs(g)) for g in gammas], dtype=torch.float32)
        self.register_buffer('phi', phi)

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Detector de resonancia simple: correlaciona hidden con patron zeta
        self.resonance_gate = nn.Sequential(
            nn.Linear(hidden_size, M),
            nn.Tanh(),
            nn.Linear(M, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs = []

        for t in range(seq_len):
            h_new, c = self.lstm_cell(x[:, t], (h, c))

            # Calcular memoria zeta
            cos_terms = torch.cos(self.gammas * t)  # type: ignore[operator, arg-type]
            zeta_weight = (self.phi * cos_terms).sum() / self.M  # type: ignore[operator]
            m_raw = zeta_weight * h

            # Gate basado en estado actual (detecta si es relevante)
            gate = self.resonance_gate(h_new)  # [batch, 1]

            # Aplicar memoria gateada
            h = h_new + gate * m_raw

            outputs.append(h)

        return torch.stack(outputs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))


if __name__ == '__main__':
    print('='*70)
    print('Zeta Resonance Detector - Test')
    print('='*70)

    # Test basico
    batch_size = 4
    seq_len = 50
    input_size = 1
    hidden_size = 32

    x = torch.randn(batch_size, seq_len, input_size)

    # Test modelo completo
    model = ZetaLSTMResonant(input_size, hidden_size, M=15)
    output, _ = model(x, return_diagnostics=True)

    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')

    diag = model.get_diagnostics()
    print(f'Avg resonance: {diag["avg_resonance"]:.4f}')
    print(f'Avg tension: {diag["avg_tension"]:.4f}')

    # Test modelo simple
    model_simple = ZetaLSTMResonantSimple(input_size, hidden_size, M=15)
    output_simple, _ = model_simple(x)
    print(f'\nSimple model output shape: {output_simple.shape}')

    # Contar parametros
    params_full = sum(p.numel() for p in model.parameters())
    params_simple = sum(p.numel() for p in model_simple.parameters())
    print(f'\nFull model params: {params_full}')
    print(f'Simple model params: {params_simple}')

    print('\n[OK] Tests passed!')
