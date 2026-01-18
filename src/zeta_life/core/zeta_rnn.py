"""
ZetaRNN: RNN layers enriched with Riemann zeta kernel memory.

Based on: "IA Adaptativa a traves de la Hipotesis de Riemann" by Francisco Ruiz

Core formula: h'_t = h_t + m_t
Where: m_t = (1/N) * sum_j(phi(gamma_j) * h_{t-1} * cos(gamma_j * t))

The zeta zeros gamma_j create oscillators that capture long-range temporal dependencies.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

# Reuse existing zeta zeros function
try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def get_zeta_zeros(M: int) -> list[float]:
    """Get first M non-trivial zeros of Riemann zeta function."""
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
                 67.079811, 69.546402, 72.067158, 75.704691, 77.144840]
        if M <= len(known):
            return known[:M]
        return known + [2 * np.pi * n / np.log(n + 2) for n in range(len(known) + 1, M + 1)]


class ZetaMemoryLayer(nn.Module):
    """
    Zeta-based temporal memory layer.

    Computes m_t = (1/N) * sum_j(phi(gamma_j) * h * cos(gamma_j * t))

    Where:
    - gamma_j: j-th non-trivial zero of Riemann zeta function
    - phi(gamma_j) = exp(-sigma * |gamma_j|): Abel regularization weight
    - t: current timestep
    - h: hidden state

    Args:
        hidden_size: Dimension of hidden state
        M: Number of zeta zeros to use (default: 15)
        sigma: Abel regularization parameter (default: 0.1)
        learnable_phi: If True, phi weights are learnable (default: False)
    """

    def __init__(
        self,
        hidden_size: int,
        M: int = 15,
        sigma: float = 0.1,
        learnable_phi: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M
        self.sigma = sigma

        # Get zeta zeros
        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))

        # Compute phi weights: phi(gamma) = exp(-sigma * |gamma|)
        phi_init = np.array([np.exp(-sigma * abs(g)) for g in gammas])

        if learnable_phi:
            self.phi = nn.Parameter(torch.tensor(phi_init, dtype=torch.float32))
        else:
            self.register_buffer('phi', torch.tensor(phi_init, dtype=torch.float32))

    def forward(self, h: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute zeta memory term m_t.

        Args:
            h: Hidden state (batch, hidden_size)
            t: Current timestep (integer)

        Returns:
            m_t: Memory contribution (batch, hidden_size)
        """
        # Compute cos(gamma_j * t) for all zeros
        # Shape: (M,)
        # Use .data to get the underlying tensor for arithmetic operations
        gammas_tensor: torch.Tensor = self.gammas  # type: ignore[assignment]
        cos_terms = torch.cos(gammas_tensor * t)

        # Weighted sum: sum_j(phi_j * cos(gamma_j * t))
        # Shape: scalar
        weights = (self.phi * cos_terms).sum() / self.M

        # Apply to hidden state
        # m_t = weights * h
        m_t = weights * h

        return m_t


class ZetaLSTMCell(nn.Module):
    """
    LSTM cell enriched with zeta temporal memory.

    Standard LSTM:
        i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
        f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
        g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
        o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    ZetaLSTM addition (from paper Section 6.2):
        m_t = ZetaMemory(h_{t-1}, t)
        h'_t = h_t + m_t

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        M: Number of zeta zeros (default: 15)
        sigma: Abel regularization (default: 0.1)
        zeta_weight: Scaling factor for zeta memory (default: 0.1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        M: int = 15,
        sigma: float = 0.1,
        zeta_weight: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zeta_weight = zeta_weight

        # Standard LSTM parameters
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Zeta memory layer
        self.zeta_memory = ZetaMemoryLayer(hidden_size, M=M, sigma=sigma)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor] | None,
        t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ZetaLSTMCell.

        Args:
            x: Input tensor (batch, input_size)
            hc: Tuple of (h, c) or None for zero initialization
            t: Current timestep

        Returns:
            (h', c): Enhanced hidden state and cell state
        """
        batch_size = x.shape[0]

        # Initialize states if needed
        if hc is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hc

        # Standard LSTM step
        h_new, c_new = self.lstm_cell(x, (h, c))

        # Zeta memory enhancement: h'_t = h_t + alpha * m_t
        m_t = self.zeta_memory(h, t)
        h_enhanced = h_new + self.zeta_weight * m_t

        return h_enhanced, c_new


class ZetaLSTM(nn.Module):
    """
    Full ZetaLSTM layer for sequence processing.

    Processes sequences through ZetaLSTMCells, adding zeta temporal
    memory at each timestep to capture long-range dependencies.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        num_layers: Number of stacked LSTM layers (default: 1)
        M: Number of zeta zeros (default: 15)
        sigma: Abel regularization (default: 0.1)
        zeta_weight: Scaling for zeta memory (default: 0.1)
        batch_first: If True, input is (batch, seq, feature) (default: True)
        dropout: Dropout between layers (default: 0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        M: int = 15,
        sigma: float = 0.1,
        zeta_weight: float = 0.1,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(
                ZetaLSTMCell(layer_input_size, hidden_size, M, sigma, zeta_weight)
            )

        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through sequence.

        Args:
            x: Input sequence (batch, seq, input) if batch_first else (seq, batch, input)
            hc: Optional initial states (h_0, c_0) each of shape (num_layers, batch, hidden)

        Returns:
            output: Sequence of hidden states (batch, seq, hidden)
            (h_n, c_n): Final states for each layer
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, input)

        seq_len, batch_size, _ = x.shape

        # Initialize states
        if hc is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [hc[0][i] for i in range(self.num_layers)]
            c = [hc[1][i] for i in range(self.num_layers)]

        # Process sequence
        outputs = []
        for t in range(seq_len):
            inp = x[t]

            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(inp, (h[layer], c[layer]), t)
                inp = h[layer]

                # Apply dropout between layers (not on last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    inp = self.dropout(inp)

            outputs.append(h[-1])

        # Stack outputs
        output = torch.stack(outputs, dim=0)  # (seq, batch, hidden)

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq, hidden)

        # Stack final states
        h_n = torch.stack(h, dim=0)  # (num_layers, batch, hidden)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)


class ZetaSequenceGenerator:
    """
    Generates synthetic sequences with zeta-based temporal dependencies.

    The target y_t depends on inputs at time lags corresponding to
    zeta zero frequencies, creating long-range dependencies that
    ZetaLSTM should capture better than vanilla LSTM.

    Formula:
        y_t = sum_j(w_j * x_{t-lag_j} * cos(gamma_j * t)) + noise

    Where lag_j = floor(period_j / 2) and period_j = 2*pi / gamma_j

    Args:
        seq_length: Length of sequences
        feature_dim: Dimension of input features
        M: Number of zeta zeros to use for dependencies
        sigma: Abel regularization for weights
        noise_std: Standard deviation of additive noise
    """

    def __init__(
        self,
        seq_length: int = 100,
        feature_dim: int = 8,
        M: int = 10,
        sigma: float = 0.1,
        noise_std: float = 0.1
    ):
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.M = M
        self.sigma = sigma
        self.noise_std = noise_std

        # Get zeta zeros and compute lags
        self.gammas = get_zeta_zeros(M)
        self.weights = [np.exp(-sigma * abs(g)) for g in self.gammas]

        # Compute characteristic lags from zeta zero indices
        # Use lags proportional to j to create multi-scale dependencies
        # This ensures dependencies at various timescales (short and long range)
        self.lags = [max(1, j * 5) for j in range(1, M + 1)]

        # Projection from feature_dim to scalar
        self.projection = np.random.randn(feature_dim) / np.sqrt(feature_dim)

    @overload
    def generate_batch(
        self,
        batch_size: int,
        return_numpy: Literal[False] = ...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def generate_batch(
        self,
        batch_size: int,
        return_numpy: Literal[True] = ...
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]: ...

    def generate_batch(
        self,
        batch_size: int,
        return_numpy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Generate a batch of sequences.

        Args:
            batch_size: Number of sequences to generate
            return_numpy: If True, return numpy arrays

        Returns:
            x: Input sequences (batch, seq_length, feature_dim)
            y: Target sequences (batch, seq_length, 1)
        """
        # Generate random input
        x = np.random.randn(batch_size, self.seq_length, self.feature_dim)

        # Compute targets with zeta-based dependencies
        y = np.zeros((batch_size, self.seq_length, 1))

        for t in range(self.seq_length):
            for j, (gamma, weight, lag) in enumerate(zip(self.gammas, self.weights, self.lags)):
                if t >= lag:
                    # Project input at t-lag to scalar
                    x_lagged = x[:, t - lag, :] @ self.projection
                    # Add oscillating contribution
                    y[:, t, 0] += weight * x_lagged * np.cos(gamma * t)

        # Add noise
        y += self.noise_std * np.random.randn(*y.shape)

        # Normalize
        y = (y - y.mean()) / (y.std() + 1e-8)

        if return_numpy:
            return x, y

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_max_lag(self) -> int:
        """Return maximum lag used in sequence generation."""
        return max(self.lags)


class ZetaLSTMExperiment:
    """
    Experiment comparing ZetaLSTM vs vanilla LSTM on zeta-correlated sequences.

    This validates the paper's conjecture that ZetaLSTM should achieve
    ~10% better performance on sequences with zeta-based dependencies.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden dimension
        seq_length: Sequence length for training
        M: Number of zeta zeros
        sigma: Abel regularization
        zeta_weight: Weight of zeta memory term
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        seq_length: int = 100,
        M: int = 15,
        sigma: float = 0.1,
        zeta_weight: float = 0.1
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.M = M
        self.sigma = sigma
        self.zeta_weight = zeta_weight

        # Create models
        self.vanilla_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.zeta_lstm = ZetaLSTM(input_size, hidden_size, M=M, sigma=sigma,
                                   zeta_weight=zeta_weight)

        # Output layers
        self.vanilla_out = nn.Linear(hidden_size, 1)
        self.zeta_out = nn.Linear(hidden_size, 1)

        # Data generator
        self.generator = ZetaSequenceGenerator(
            seq_length=seq_length,
            feature_dim=input_size,
            M=M,
            sigma=sigma
        )

    def _train_epoch(
        self,
        model: nn.Module,
        output_layer: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_batches: int
    ) -> float:
        """Train one epoch."""
        model.train()
        output_layer.train()
        total_loss = 0.0

        for _ in range(num_batches):
            x, y = self.generator.generate_batch(batch_size)

            optimizer.zero_grad()

            output, _ = model(x)
            pred = output_layer(output)

            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def _eval(
        self,
        model: nn.Module,
        output_layer: nn.Module,
        batch_size: int,
        num_batches: int
    ) -> float:
        """Evaluate model."""
        model.eval()
        output_layer.eval()
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(num_batches):
                x, y = self.generator.generate_batch(batch_size)
                output, _ = model(x)
                pred = output_layer(output)
                loss = nn.functional.mse_loss(pred, y)
                total_loss += loss.item()

        return total_loss / num_batches

    def run(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        batches_per_epoch: int = 20,
        lr: float = 1e-3
    ) -> dict[str, Any]:
        """
        Run comparison experiment.

        Returns:
            Dictionary with training curves and final metrics
        """
        # Optimizers
        vanilla_params = list(self.vanilla_lstm.parameters()) + list(self.vanilla_out.parameters())
        zeta_params = list(self.zeta_lstm.parameters()) + list(self.zeta_out.parameters())

        vanilla_opt = torch.optim.Adam(vanilla_params, lr=lr)
        zeta_opt = torch.optim.Adam(zeta_params, lr=lr)

        results: dict[str, Any] = {
            'vanilla_loss': [],
            'zeta_loss': [],
            'vanilla_eval': [],
            'zeta_eval': []
        }

        for epoch in range(epochs):
            # Train
            v_loss = self._train_epoch(self.vanilla_lstm, self.vanilla_out,
                                       vanilla_opt, batch_size, batches_per_epoch)
            z_loss = self._train_epoch(self.zeta_lstm, self.zeta_out,
                                       zeta_opt, batch_size, batches_per_epoch)

            results['vanilla_loss'].append(v_loss)
            results['zeta_loss'].append(z_loss)

            # Eval
            v_eval = self._eval(self.vanilla_lstm, self.vanilla_out, batch_size, 5)
            z_eval = self._eval(self.zeta_lstm, self.zeta_out, batch_size, 5)

            results['vanilla_eval'].append(v_eval)
            results['zeta_eval'].append(z_eval)

        # Compute improvement
        final_vanilla = float(np.mean(results['vanilla_eval'][-5:]))
        final_zeta = float(np.mean(results['zeta_eval'][-5:]))
        improvement = (final_vanilla - final_zeta) / final_vanilla * 100

        results['final_vanilla_loss'] = final_vanilla
        results['final_zeta_loss'] = final_zeta
        results['improvement_percent'] = improvement

        return results


def demo_zeta_lstm():
    """
    Full demonstration of ZetaLSTM vs vanilla LSTM.

    Validates paper conjecture on ~10% improvement for zeta-correlated sequences.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("ZETA LSTM EXPERIMENT")
    print("Validating: h'_t = h_t + m_t from 'IA Adaptativa via Riemann'")
    print("=" * 70)

    # 1. Setup experiment
    print("\n1. Creating experiment...")
    exp = ZetaLSTMExperiment(
        input_size=8,
        hidden_size=64,
        seq_length=100,
        M=15,
        sigma=0.1,
        zeta_weight=0.1
    )

    print(f"   Input size: {exp.input_size}")
    print(f"   Hidden size: {exp.hidden_size}")
    print(f"   Sequence length: {exp.seq_length}")
    print(f"   Zeta zeros (M): {exp.M}")
    print(f"   Zeta weight: {exp.zeta_weight}")

    # 2. Run experiment
    print("\n2. Running experiment (50 epochs)...")
    results = exp.run(epochs=50, batch_size=32, batches_per_epoch=20)

    # 3. Report results
    print("\n3. Results:")
    print(f"   Vanilla LSTM final loss: {results['final_vanilla_loss']:.6f}")
    print(f"   Zeta LSTM final loss:    {results['final_zeta_loss']:.6f}")
    print(f"   Improvement: {results['improvement_percent']:.2f}%")

    if results['improvement_percent'] > 0:
        print(f"\n   [OK] ZetaLSTM outperforms vanilla by {results['improvement_percent']:.1f}%")
        if results['improvement_percent'] >= 10:
            print("   [OK] Paper conjecture (~10% improvement) VALIDATED!")
    else:
        print("\n   [--] No improvement observed")

    # 4. Plot results
    print("\n4. Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(results['vanilla_loss'], label='Vanilla LSTM', color='blue')
    axes[0].plot(results['zeta_loss'], label='Zeta LSTM', color='green')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Eval loss
    axes[1].plot(results['vanilla_eval'], label='Vanilla LSTM', color='blue')
    axes[1].plot(results['zeta_eval'], label='Zeta LSTM', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Evaluation Loss')
    axes[1].set_title(f'Evaluation Loss (Improvement: {results["improvement_percent"]:.1f}%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_lstm_experiment.png', dpi=150, bbox_inches='tight')
    print("   Saved: zeta_lstm_experiment.png")

    # 5. Visualize zeta memory oscillation
    print("\n5. Visualizing zeta memory kernel...")

    fig2, ax = plt.subplots(figsize=(10, 4))

    memory_layer = ZetaMemoryLayer(hidden_size=64, M=15, sigma=0.1)
    h = torch.ones(1, 64)

    t_vals = list(range(100))
    m_vals = [memory_layer(h, t)[0, 0].item() for t in t_vals]

    ax.plot(t_vals, m_vals, 'g-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('m_t (memory contribution)')
    ax.set_title('Zeta Memory Oscillation Pattern (based on Riemann zeros)')
    ax.grid(True, alpha=0.3)

    plt.savefig('zeta_memory_oscillation.png', dpi=150, bbox_inches='tight')
    print("   Saved: zeta_memory_oscillation.png")

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    demo_zeta_lstm()
