# Plan: planificación tipo Dreamer (actor-crítico por imaginación latente)

**Fecha:** 2026-06-08
**Objetivo:** reemplazar la selección de acción por búsqueda (shooting/CEM) por un
**actor amortizado entrenado en imaginación** con un crítico y gradientes de valor
a través de la dinámica latente diferenciable — el patrón Dreamer adaptado a la
inferencia activa del kernel. Es el upgrade #1 de `docs/RELATED_WORK.md`.

## Motivación (de los resultados de la sesión)

- El planner actual hace *shooting/CEM* de 1 paso: caro por step, no escala
  (`exp_cem.py` mostró que CEM no ayuda en este régimen; el costo por step crece
  con `efe_n_samples`).
- El kernel ya mostró ventaja en **control model-based** (`exp_modelcontrol.py`):
  aprende dinámica desconocida y la invierte. Un actor amortizado debería lograr
  ese control a **costo constante por step** y **escalar a más dimensión**.
- La literatura (Dreamer, Hafner et al.) es el SOTA: imaginación latente +
  actor-crítico + gradientes de valor. Lo adaptamos usando **−EFE como reward**.

## Idea central

Reward de inferencia activa por step imaginado:

```
r(z) = − KL( C ‖ simplex(predictor(z)) )   [pragmático]
       + w · disagreement(z)               [epistémico, opcional]
```

- **Actor** `π(z) → acción` (logits → softmax = punto del símplex; determinista
  para gradientes de valor, con ruido de exploración en warmup).
- **Crítico** `V(z) → valor` (retorno esperado acumulado de r).
- **Imaginación:** desde el latente actual, rollout diferenciable de H pasos bajo
  el actor; se calculan rewards y **λ-returns**; el crítico se entrena a predecir
  los λ-returns; el actor se entrena a **maximizar** los λ-returns propagando
  gradientes de valor por la dinámica latente.
- **En acción:** el actor produce la acción directamente — **sin búsqueda**.

## Decisiones de diseño (y por qué)

1. **Actor determinista sobre el símplex** (`softmax(logits)`), no estocástico:
   permite gradientes de valor directos (estilo Dreamer-continuo / DDPG) sin
   Gumbel-softmax. Exploración = ruido en logits durante warmup. *(Alternativa
   descartada para v1: política estocástica + REINFORCE — mayor varianza.)*
2. **Reward = −EFE** (no un reward externo): mantiene la coherencia con la
   inferencia activa. Pragmático siempre; epistémico (disagreement) detrás de
   flag (su efecto ya lo sabemos frágil; lo dejamos opcional).
3. **Dinámica latente diferenciable:** `imagine()` actual es `@no_grad`. Hace
   falta un rollout con gradiente (`imagine_grad`). El `GRUCell` ya es
   diferenciable; solo hay que no cortar el grafo.
4. **Online con mini-replay de latentes:** Dreamer usa replay buffer. El kernel
   es online; agregamos un buffer chico de latentes recientes y, cada step,
   hacemos K rollouts de imaginación desde latentes muestreados para entrenar
   actor/crítico. Mantiene el espíritu online sin colapsar la diversidad.
5. **Backward-compat total:** nuevo `action_mode="dreamer"`. Los modos
   `reactive`/`efe` quedan intactos; defaults byte-idénticos.
6. **Preferencia C fija** en v1 (igual que el EFE actual). Target móvil = futuro.

## Fases

### Fase 0 — Andamiaje de evaluación (primero el criterio)
- Definir el benchmark de comparación **antes** de implementar:
  - Tareas: control model-based permutado (`exp_modelcontrol.py`) + una tarea de
    **mayor dimensión** (obs_dim ∈ {4, 8, 16}) para testear escala.
  - Métricas: (a) error de control `‖estado−C‖`; (b) **costo por step** (tiempo /
    nº de `imagine`); (c) escala (error vs dimensión).
  - Baselines a batir/igualar: EFE-shooting (`efe_n_samples=48`), EFE-CEM.
- **Criterio de éxito:** el actor amortizado **iguala** (±10%) el error de control
  de shooting/CEM, a **costo por step constante** (no crece con la dimensión como
  el shooting), y **escala mejor** a obs_dim alto.
- **Criterio de falsación (honesto):** si el actor no alcanza el error de
  shooting, o no entrena de forma estable, se reporta como negativo y se
  documenta por qué (igual que CEM/curiosidad).

### Fase 1 — Dinámica latente diferenciable (`world_model.py`)
- Añadir `imagine_grad(self, actions, temporal_feats=None) -> (latents, preds)`
  que hace el rollout SIN `no_grad`, devolviendo latentes y predicciones con
  grafo. No tocar `imagine()` (lo usa el EFE).
- Tests: shapes correctas; los tensores tienen `requires_grad`; `imagine()`
  (no-grad) sigue intacto.

### Fase 2 — Actor y Crítico (`kernel/policy.py`, nuevo)
- `Actor(nn.Module)`: `Linear(latent_dim, h) → ReLU → Linear(h, obs_dim)`;
  `forward(z) -> softmax(logits)` (acción en el símplex).
- `Critic(nn.Module)`: `Linear(latent_dim, h) → ReLU → Linear(h, 1)`.
- Optimizers separados (actor_lr, critic_lr).
- Tests: shapes; acción suma 1; valor escalar.

### Fase 3 — Aprendizaje por imaginación (`conscious_kernel.py`)
- Nuevos params: `action_mode="dreamer"`, `imag_horizon` (H, def 5),
  `imag_rollouts` (K, def 4), `imag_lambda` (λ, def 0.95), `imag_gamma`
  (γ, def 0.97), `actor_lr`, `critic_lr`, `actor_explore` (ruido warmup),
  `dreamer_epistemic_weight` (def 0).
- Buffer de latentes recientes (deque pequeño).
- `_reward(z) -> Tensor`: `−KL(C ‖ simplex(predictor(z)))` (+ epistémico opcional).
- `_train_behavior()` (llamado en `step()` tras `observe()` cuando
  `action_mode=="dreamer"`):
  1. muestrear latentes de inicio del buffer;
  2. `imagine_grad` H pasos bajo el actor;
  3. rewards por step + bootstrap con el crítico → **λ-returns**;
  4. critic loss = MSE(V, λ-returns.detach()); step;
  5. actor loss = −mean(λ-returns) (value gradients por la dinámica); step.
- `_select_action_dreamer()`: `actor(latent_state)` (+ ruido si warmup).
- Wire en `step()`: rama `action_mode=="dreamer"`.
- Tests: el kernel corre en modo dreamer; acción válida; actor/critic params se
  mueven; defaults intactos (full suite verde).

### Fase 4 — Experimento y evaluación honesta (`experiments/kernel/exp_dreamer.py`)
- Comparar **dreamer (amortizado)** vs **efe-shooting** vs **efe-cem** en:
  - control permutado (mismo target no-vértice);
  - barrido de dimensión obs_dim ∈ {4, 8, 16} → error + costo por step.
- Reportar: error de control, tiempo/step, y la curva error-vs-dimensión.
- Veredicto honesto contra los criterios de Fase 0. Figura + run.txt.

### Fase 5 — Documentación y cierre
- Paper: nueva subsección (§2.5 actualizado + un §3.x con el resultado) y entrada
  en el ledger (sí/no según los datos).
- CLAUDE.md (params + experimento), RELATED_WORK (marcar el upgrade hecho).
- Tests nuevos en la suite; commit por fase o uno final; merge/push.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Inestabilidad de actor-crítico (divergencia de valor) | normalización de returns, lr bajos, clip de gradiente; empezar con H corto |
| Gradientes de valor por GRU explotan/desvanecen | grad clip; H≤5 al inicio; symlog/normalización de reward estilo DreamerV3 |
| Reward = −KL mal escalado | normalizar reward por EMA (como Ψ/precisiones); documentarlo |
| Buffer online colapsa diversidad | buffer de latentes con muestreo; warmup exploratorio |
| Otro negativo honesto (no iguala al shooting) | está previsto: se reporta como tal; el costo-por-step constante puede ser el aporte aunque el error empate |
| Romper modos existentes | `action_mode` nuevo; defaults intactos; full suite verde por fase |

## Alcance explícito (lo que NO entra en v1)
- RSSM estocástico (mantenemos el GRU determinista actual).
- Target móvil / preferencia-trayectoria (preferencia fija).
- Benchmark RL externo (gym/DMC) — fase posterior, una vez el actor funcione.

## Estimación
~5 fases, incremental con TDD. Cada fase deja la suite verde y es commiteable por
separado. El núcleo (Fases 1–3) es el grueso; Fase 4 decide el veredicto.
