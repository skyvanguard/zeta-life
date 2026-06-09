# Plan: paridad DreamerV3 — RSSM + reward model (¿crackea CartPole?)

**Fecha:** 2026-06-08
**Objetivo:** construir un agente **DreamerV2/V3-style** con un **RSSM** (modelo
recurrente de estado-espacio entrenado sobre **secuencias**) y un **reward model
aprendido**, y medir si alcanza el techo de CartPole-v1 (~500). Esto **acota** la
limitación del kernel: si un RSSM propio crackea CartPole donde el world-model de
un paso del kernel se estanca en ~33%, el cuello de botella es la arquitectura del
world-model, no el diseño de inferencia activa.

## Por qué (del resultado previo)

El loop dreamer del kernel (replay de transiciones de **un paso** + reward fijo
`neg_distance`) sube a un pico ~50% del techo en CartPole y luego colapsa; el eval
greedy se estanca en ~33%. La conclusión honesta fue: cerrar la brecha requiere
(a) un world-model **recurrente entrenado sobre secuencias** (estado latente con
memoria temporal real + imaginación multi-paso precisa) y (b) un **reward model
aprendido** (no un proxy de distancia a un goal fijo).

## Alcance

Agente **autocontenido** (no toca `ConsciousKernel`; preserva los 504 tests). Es
una **implementación de referencia** para acotar el kernel, no un reemplazo. Latente
estocástico **Gaussiano** (V2-flavored) con KL balancing + free nats — suficiente
para CartPole; honestamente NO es el categórico/two-hot/symlog verbatim de V3.

## Componentes

### `kernel/rssm.py` — el world model recurrente
- `encoder(obs) -> embed`
- recurrente: `h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})` (estado determinista)
- **prior** `p(z_t | h_t)` y **posterior** `q(z_t | h_t, embed_t)` (Gaussianas)
- feature `s_t = [h_t, z_t]`
- cabezas desde `s_t`: **decoder** (recon obs), **reward** (symlog MSE), **continue** (BCE)
- `observe(obs_seq, act_seq)`: BPTT sobre la secuencia → estados posteriores + pérdidas
  (recon + reward + continue + **KL balanceada** con free nats)
- `imagine_step(s, a)`: un paso usando el **prior** (sin obs) → `s'`, reward, continue

### `kernel/dreamerv3_agent.py` — replay de secuencias + actor-crítico
- `SequenceReplay`: guarda episodios; muestrea B secuencias de largo L
- `Actor(s) -> Categorical(logits)`, `Critic(s) -> valor` (target EMA)
- `train_world_model()`: muestrea secuencias → `rssm.observe` → step del optimizer del WM
- `train_behavior()`: desde los estados posteriores (flatten B×L), **imaginación** de H
  pasos bajo el actor (sampling); **λ-returns** con el reward/continue aprendidos;
  crítico regresa a λ-returns; actor por **reinforce + baseline + entropía** (acción discreta)
- `act(obs)`: paso del estado (GRU+posterior con la obs real) → `actor` (sample/greedy)

### `experiments/kernel/exp_dreamerv3.py`
- entrena en CartPole-v1; baselines random/heurística; compara vs el actor amortizado
  del kernel (`exp_cartpole.py`); métrica = largo de episodio (cap 500); eval greedy
- veredicto: ¿alcanza el techo (≥475)? ¿supera el ~33% del kernel?

## Criterios

- **Éxito:** el RSSM **resuelve CartPole** (eval ≥ ~450/500) → el cuello de botella del
  kernel era su world-model de un paso (resultado fuerte, acota la limitación).
- **Parcial/Falla honesta:** si el RSSM tampoco lo resuelve, el límite es más profundo
  (capacidad/entorno/implementación) — se reporta como tal, sin tunear sin fin.

## Fases (incremental, TDD, suite verde por fase)
1. `rssm.py` + tests (shapes, observe/imagine devuelven estados y pérdidas finitas)
2. `SequenceReplay` + Actor/Critic + tests
3. `dreamerv3_agent.py` train loop (WM + behavior) + tests (corre, params se mueven)
4. `exp_dreamerv3.py` en CartPole + veredicto honesto + figura
5. docs (paper §, CLAUDE, RELATED_WORK), commit

## Riesgos
| Riesgo | Mitigación |
|---|---|
| RL finicky (no converge) | hiperparámetros known-good para CartPole-escala; testear cada pieza; horizonte/entropía/lr conservadores |
| Inestabilidad de KL (posterior collapse) | KL balancing 0.8 + free nats 1.0 |
| Costo de cómputo (BPTT secuencias) | L=32, B=16, redes chicas (deter 128, stoch 32); CartPole es liviano |
| Otra falla honesta | prevista: se reporta; el valor es acotar dónde está el límite |
