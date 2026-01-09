# exp_zeta_lstm_memory_test.py
"""
Test directo de memoria temporal: Tarea de eco con delays zeta.

La tarea es simple: el modelo debe recordar un valor de entrada
y reproducirlo despues de un delay especifico.

Los delays corresponden a periodos de los ceros zeta:
- gamma_1 = 14.13 -> periodo ~0.44 -> lag ~22 steps
- gamma_2 = 21.02 -> periodo ~0.30 -> lag ~15 steps
- etc.

ZetaLSTM deberia tener ventaja porque su memoria oscila
naturalmente a estas frecuencias.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from zeta_life.core import ZetaLSTM, get_zeta_zeros


class ZetaEchoTask:
    """
    Tarea de eco con delays basados en ceros zeta.

    Input: Secuencia con pulsos en posiciones aleatorias
    Output: Eco de los pulsos despues de delay zeta
    """

    def __init__(self, seq_length=100, delay_mode='single'):
        self.seq_length = seq_length
        self.delay_mode = delay_mode

        # Delays basados en periodos zeta
        gammas = get_zeta_zeros(5)
        self.zeta_delays = [max(5, int(2 * np.pi / g * 5)) for g in gammas]
        print(f"Zeta delays: {self.zeta_delays}")

        if delay_mode == 'single':
            self.delay = self.zeta_delays[0]  # Usar primer delay zeta
        else:
            self.delay = self.zeta_delays

    def generate_batch(self, batch_size):
        """Genera batch de secuencias eco."""
        x = np.zeros((batch_size, self.seq_length, 1))
        y = np.zeros((batch_size, self.seq_length, 1))

        for b in range(batch_size):
            # Colocar pulsos aleatorios
            n_pulses = np.random.randint(3, 8)
            pulse_positions = np.random.choice(
                range(10, self.seq_length - max(self.zeta_delays) - 10),
                size=n_pulses,
                replace=False
            )
            pulse_values = np.random.randn(n_pulses)

            for pos, val in zip(pulse_positions, pulse_values):
                x[b, pos, 0] = val

                if self.delay_mode == 'single':
                    echo_pos = pos + self.delay
                    if echo_pos < self.seq_length:
                        y[b, echo_pos, 0] = val
                else:
                    # Multiple ecos con diferentes delays
                    for d in self.delay:
                        echo_pos = pos + d
                        if echo_pos < self.seq_length:
                            y[b, echo_pos, 0] += val * 0.5

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ZetaAdditionTask:
    """
    Tarea de suma con marcadores: recordar dos numeros y sumarlos.

    Clasico benchmark para memoria de largo alcance.
    Los marcadores aparecen en posiciones con separacion zeta.
    """

    def __init__(self, seq_length=100):
        self.seq_length = seq_length
        gammas = get_zeta_zeros(3)
        self.zeta_period = int(2 * np.pi / gammas[0] * 5)
        print(f"Zeta period for addition task: {self.zeta_period}")

    def generate_batch(self, batch_size):
        """
        Input: (seq_length, 2)
          - Canal 0: numeros aleatorios
          - Canal 1: marcadores (1 en posiciones a recordar)
        Output: Suma de los dos numeros marcados (al final)
        """
        x = np.zeros((batch_size, self.seq_length, 2))
        y = np.zeros((batch_size, 1))

        for b in range(batch_size):
            # Numeros aleatorios en todo el canal 0
            x[b, :, 0] = np.random.uniform(0, 1, self.seq_length)

            # Dos marcadores separados por periodo zeta
            pos1 = np.random.randint(5, 20)
            pos2 = pos1 + self.zeta_period

            if pos2 < self.seq_length - 10:
                x[b, pos1, 1] = 1.0
                x[b, pos2, 1] = 1.0
                y[b, 0] = x[b, pos1, 0] + x[b, pos2, 0]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_and_evaluate(model_name, model, output_layer, task, epochs=100, lr=1e-3):
    """Entrena y evalua un modelo."""
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_layer.parameters()),
        lr=lr
    )

    losses = []

    for epoch in range(epochs):
        model.train()
        x, y = task.generate_batch(32)

        optimizer.zero_grad()
        out, _ = model(x)

        # Para tarea de eco: usar toda la secuencia
        # Para tarea de suma: usar ultimo timestep
        if isinstance(task, ZetaAdditionTask):
            pred = output_layer(out[:, -1, :])
        else:
            pred = output_layer(out)

        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            x, y = task.generate_batch(64)
            out, _ = model(x)
            if isinstance(task, ZetaAdditionTask):
                pred = output_layer(out[:, -1, :])
            else:
                pred = output_layer(out)
            eval_loss = nn.functional.mse_loss(pred, y).item()

        losses.append(eval_loss)

        if (epoch + 1) % 25 == 0:
            print(f"  {model_name} Epoch {epoch+1}: Loss={eval_loss:.4f}")

    return losses


def run_echo_test():
    """Test de eco con delay zeta."""
    print("\n" + "="*70)
    print("TEST 1: TAREA DE ECO CON DELAY ZETA")
    print("="*70)

    task = ZetaEchoTask(seq_length=100, delay_mode='single')
    print(f"Delay: {task.delay} steps")

    hidden_size = 64

    # Vanilla LSTM
    print("\nEntrenando Vanilla LSTM...")
    vanilla = nn.LSTM(1, hidden_size, batch_first=True)
    vanilla_out = nn.Linear(hidden_size, 1)
    vanilla_losses = train_and_evaluate("Vanilla", vanilla, vanilla_out, task, epochs=150)

    # Zeta LSTM
    print("\nEntrenando Zeta LSTM...")
    zeta = ZetaLSTM(1, hidden_size, M=10, zeta_weight=0.3)
    zeta_out = nn.Linear(hidden_size, 1)
    zeta_losses = train_and_evaluate("Zeta", zeta, zeta_out, task, epochs=150)

    # Resultados
    final_v = np.mean(vanilla_losses[-10:])
    final_z = np.mean(zeta_losses[-10:])
    improvement = (final_v - final_z) / final_v * 100 if final_v > 0 else 0

    print(f"\nResultados Echo Test:")
    print(f"  Vanilla: {final_v:.4f}")
    print(f"  Zeta:    {final_z:.4f}")
    print(f"  Mejora:  {improvement:+.1f}%")

    return vanilla_losses, zeta_losses, improvement


def run_addition_test():
    """Test de suma con separacion zeta."""
    print("\n" + "="*70)
    print("TEST 2: TAREA DE SUMA CON SEPARACION ZETA")
    print("="*70)

    task = ZetaAdditionTask(seq_length=80)

    hidden_size = 64

    # Vanilla LSTM
    print("\nEntrenando Vanilla LSTM...")
    vanilla = nn.LSTM(2, hidden_size, batch_first=True)
    vanilla_out = nn.Linear(hidden_size, 1)
    vanilla_losses = train_and_evaluate("Vanilla", vanilla, vanilla_out, task, epochs=200)

    # Zeta LSTM
    print("\nEntrenando Zeta LSTM...")
    zeta = ZetaLSTM(2, hidden_size, M=10, zeta_weight=0.3)
    zeta_out = nn.Linear(hidden_size, 1)
    zeta_losses = train_and_evaluate("Zeta", zeta, zeta_out, task, epochs=200)

    # Resultados
    final_v = np.mean(vanilla_losses[-10:])
    final_z = np.mean(zeta_losses[-10:])
    improvement = (final_v - final_z) / final_v * 100 if final_v > 0 else 0

    print(f"\nResultados Addition Test:")
    print(f"  Vanilla: {final_v:.4f}")
    print(f"  Zeta:    {final_z:.4f}")
    print(f"  Mejora:  {improvement:+.1f}%")

    return vanilla_losses, zeta_losses, improvement


def run_variable_delay_test():
    """Test con delays variables para ver donde ZetaLSTM tiene ventaja."""
    print("\n" + "="*70)
    print("TEST 3: BARRIDO DE DELAYS")
    print("Probando delays de 5 a 40 steps")
    print("="*70)

    gammas = get_zeta_zeros(5)
    zeta_periods = [int(2 * np.pi / g * 5) for g in gammas]
    print(f"Periodos zeta naturales: {zeta_periods}")

    delays_to_test = [5, 10, 15, 20, 25, 30, 35, 40]
    results = []

    for delay in delays_to_test:
        print(f"\n--- Delay = {delay} ---")

        # Custom task con delay especifico
        class FixedDelayTask:
            def __init__(self, d):
                self.delay = d
                self.seq_length = 80

            def generate_batch(self, batch_size):
                x = np.zeros((batch_size, self.seq_length, 1))
                y = np.zeros((batch_size, self.seq_length, 1))

                for b in range(batch_size):
                    n_pulses = 5
                    positions = np.random.choice(range(5, 30), n_pulses, replace=False)
                    values = np.random.randn(n_pulses)

                    for pos, val in zip(positions, values):
                        x[b, pos, 0] = val
                        echo_pos = pos + self.delay
                        if echo_pos < self.seq_length:
                            y[b, echo_pos, 0] = val

                return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        task = FixedDelayTask(delay)
        hidden = 32

        # Quick train
        vanilla = nn.LSTM(1, hidden, batch_first=True)
        vanilla_out = nn.Linear(hidden, 1)
        v_losses = train_and_evaluate("V", vanilla, vanilla_out, task, epochs=80, lr=2e-3)

        zeta = ZetaLSTM(1, hidden, M=5, zeta_weight=0.3)
        zeta_out = nn.Linear(hidden, 1)
        z_losses = train_and_evaluate("Z", zeta, zeta_out, task, epochs=80, lr=2e-3)

        final_v = np.mean(v_losses[-5:])
        final_z = np.mean(z_losses[-5:])
        imp = (final_v - final_z) / final_v * 100 if final_v > 0 else 0

        is_zeta_period = delay in zeta_periods or any(abs(delay - p) <= 2 for p in zeta_periods)

        results.append({
            'delay': delay,
            'vanilla': final_v,
            'zeta': final_z,
            'improvement': imp,
            'is_zeta': is_zeta_period
        })

        marker = " <-- ZETA PERIOD" if is_zeta_period else ""
        print(f"  Delay {delay}: V={final_v:.4f}, Z={final_z:.4f}, Mejora={imp:+.1f}%{marker}")

    return results


def main():
    print("="*70)
    print("VALIDACION DE MEMORIA TEMPORAL: ZetaLSTM")
    print("Tests directos de capacidad de memoria de largo alcance")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Test 1: Echo
    echo_v, echo_z, echo_imp = run_echo_test()

    # Test 2: Addition
    add_v, add_z, add_imp = run_addition_test()

    # Test 3: Variable delay
    delay_results = run_variable_delay_test()

    # Visualizacion
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Echo test
    ax = axes[0, 0]
    ax.plot(echo_v, 'b-', label='Vanilla', alpha=0.7)
    ax.plot(echo_z, 'g-', label='Zeta', alpha=0.7)
    ax.set_title(f'Echo Test (Mejora: {echo_imp:+.1f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Addition test
    ax = axes[0, 1]
    ax.plot(add_v, 'b-', label='Vanilla', alpha=0.7)
    ax.plot(add_z, 'g-', label='Zeta', alpha=0.7)
    ax.set_title(f'Addition Test (Mejora: {add_imp:+.1f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Delay sweep - losses
    ax = axes[1, 0]
    delays = [r['delay'] for r in delay_results]
    v_losses = [r['vanilla'] for r in delay_results]
    z_losses = [r['zeta'] for r in delay_results]

    ax.plot(delays, v_losses, 'bo-', label='Vanilla', markersize=8)
    ax.plot(delays, z_losses, 'go-', label='Zeta', markersize=8)

    # Marcar periodos zeta
    for r in delay_results:
        if r['is_zeta']:
            ax.axvline(x=r['delay'], color='orange', linestyle='--', alpha=0.5)

    ax.set_title('Loss vs Delay (lineas naranjas = periodos zeta)')
    ax.set_xlabel('Delay (steps)')
    ax.set_ylabel('Final Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Delay sweep - improvement
    ax = axes[1, 1]
    improvements = [r['improvement'] for r in delay_results]
    colors = ['green' if r['is_zeta'] else 'blue' for r in delay_results]

    bars = ax.bar(delays, improvements, color=colors, alpha=0.7, width=3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=10, color='red', linestyle='--', label='Target 10%')

    ax.set_title('Mejora de ZetaLSTM por Delay')
    ax.set_xlabel('Delay (steps)')
    ax.set_ylabel('Mejora (%)')
    ax.legend(['', 'Target 10%', 'Delay zeta', 'Otro delay'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_lstm_memory_test.png', dpi=150)
    print("\nGuardado: zeta_lstm_memory_test.png")

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"\n{'Test':<25} {'Mejora ZetaLSTM':<20}")
    print("-"*45)
    print(f"{'Echo (delay zeta)':<25} {echo_imp:+.1f}%")
    print(f"{'Addition (sep zeta)':<25} {add_imp:+.1f}%")

    # Promedio en periodos zeta vs otros
    zeta_delays_imp = [r['improvement'] for r in delay_results if r['is_zeta']]
    other_delays_imp = [r['improvement'] for r in delay_results if not r['is_zeta']]

    if zeta_delays_imp:
        print(f"{'Promedio delays zeta':<25} {np.mean(zeta_delays_imp):+.1f}%")
    if other_delays_imp:
        print(f"{'Promedio otros delays':<25} {np.mean(other_delays_imp):+.1f}%")

    # Conclusion
    best_imp = max(echo_imp, add_imp, max(improvements) if improvements else 0)

    if best_imp >= 10:
        print(f"\n*** CONJETURA VALIDADA: {best_imp:.1f}% mejora ***")
    elif best_imp > 5:
        print(f"\n*** EVIDENCIA PARCIAL: {best_imp:.1f}% mejora (< 10% esperado) ***")
    elif best_imp > 0:
        print(f"\n*** MEJORA MARGINAL: {best_imp:.1f}% ***")
    else:
        print(f"\n*** NO SE OBSERVA VENTAJA CLARA ***")


if __name__ == '__main__':
    main()
