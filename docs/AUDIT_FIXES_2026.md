# Auditoría y corrección del núcleo (2026)

Registro de la auditoría de implementación del `ConsciousKernel` y las
ecuaciones formales, y las correcciones aplicadas. **Todo fueron bugs de
implementación; la teoría (transición de fase, free energy, self-model
recursivo, kernel zeta) no se modificó.**

## Motivación

Al intentar medir Ψ como discriminador (input coherente vs ruido), Ψ saturaba a
1.0 para ambos. La investigación reveló dos patrones de fondo:
1. **Los mecanismos de aprendizaje no aprendían** (precisiones, encoder,
   self-model congelados; recurrencia descartada).
2. **Clamps y polos** que enmascaraban el comportamiento de varias fórmulas.

## Los 11 puntos corregidos

### Ecuaciones formales (Fase 1)
- **#7 `compute_B`** — polo en `phi_c→0⁺` (explotaba a ~1e9). Guard con `_PHI_C_MIN`.
- **#8 `predict_system_stability`** — régimen subcrítico (α≤C) se reportaba `STABLE`; ahora `SUBCRITICAL`.
- **#9 `compute_corruption_threshold`** — clampeaba negativos a 0 contra su docstring; ahora los preserva (señal de inestabilidad) y valida inputs.
- **#10 `compute_M_c`** — caveat dimensional documentado (masa vs información integrada).
- **#11 `micro_psyche.compute_surprise`** — L1 entre distribuciones ∈[0,2] clampeada; ahora normalizada `/2` a [0,1].

### Sistema de precisión (Fase 2)
- **#1 precisiones congeladas** — `log_precisions` nunca se entrenaba (fija en 0.693). Optimizer dedicado + `update_precisions()` cada step.
- **#2 `PrecisionController`** — código muerto (instanciado/persistido, `forward` nunca llamado). Eliminado.
- **#3 free energy de entrenamiento** — faltaba el término `−log precision`; minimizar colapsaba la precisión a 0. Corregido en `update_precisions` (óptimo = inverse variance).

### World model (Fase 3)
- **#4 encoder congelado** — `latent_state` siempre detacheado, el gradiente nunca llegaba al encoder. Nuevo `observe()` (posterior) entrena el encoder por reconstrucción.
- **#5 recurrencia destruida** — `latent_state` se sobrescribía con `encode(stimulus).detach()` cada step, descartando la transición GRU. Ahora `predict()` mantiene el prior recurrente y `observe()` aplica la corrección posterior.

### Self model (Fase 4)
- **#6 self-model congelado** — solo se backprop. el error perceptual; el canal interoceptivo era un offset fijo. Optimizer propio + `update_from_error()` (excluye `self_embedding`, que sigue por EMA).

## Impacto medido

| Métrica | Antes (congelado) | Después (aprende) |
|---|---|---|
| free_energy coherente vs ruido | 0.35 vs 0.42 (1.2×) | 0.08 vs 0.74 (**9.3×**) |
| Precisiones | fijas 0.693 | entrenadas (~4.6 en canales activos) |
| Encoder Δ tras entrenamiento | 0.0 (congelado) | 1.4–40 |
| Error interoceptivo (kernel) | plano | baja ~8× (0.046→0.006) |
| Memoria temporal (secuencia cíclica) | error 0.0216 | error 0.0014 (**15×**) |
| Validación del kernel | 5/6 | **6/6** (generalización pasó) |
| Ψ discriminación coherente/ruido | 1.0 / 1.0 (nula) | **0.998 / 0.11** |

## Decisiones

- **Métrica Ψ por defecto: Hill** (`compute_psi_hill`, `psi_mode="hill"`). La
  forma cúbica `B³+Φ` satura (clamp) y no discrimina grados de integración. La
  Hill `Bⁿ/(Kⁿ+Bⁿ)` es acotada, continua y sin clamp. La cúbica se conserva
  (`psi_mode="cubic"`) para reproducir resultados previos.
- **Hiperparámetros Hill por defecto** (calibrados sobre el sistema que aprende):
  `psi_fe_scale=5.0, psi_hill_n=4.0, psi_hill_K=0.1`. Conviene recalibrar por
  régimen/experimento.

## Pendiente (fuera de los 11; limpieza menor 🟢, no aplicada)
- Canales `temporal`/`epistemic` siguen sin señal (zeros). Activarlos daría más
  vías de free energy.
- `DreamEngine`: consolidación marginal con `lr=0.0001` (pre≈post). Subir lr o
  iterar más para efecto medible.
- Campo `weighted` en `compute_errors` calculado y no usado.

## Reproducir
```bash
PYTHONPATH=src python -m pytest -q                       # 770 tests
PYTHONPATH=src python experiments/kernel/exp_conscious_kernel_validation.py  # 6/6
```
