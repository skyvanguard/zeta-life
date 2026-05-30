# Agencia: selección de acción por active inference (2026)

Registro del trabajo que dio **agencia** al `ConsciousKernel`: pasar de un
sustrato que solo *percibe* (Ψ discrimina coherencia de ruido, ver
[AUDIT_FIXES_2026](AUDIT_FIXES_2026.md)) a uno que *actúa con propósito* hacia
una preferencia. **No se modificó la teoría ni el world model; el fix fue de
alineamiento + un selector de acción.**

## Motivación

Tras calibrar Ψ, la acción del kernel era `softmax(stimulus)` — puramente
reactiva. El parámetro `latent_weight` (que mezclaba un bias del estado latente)
**no aportaba nada**: barrido de 0.0 a 2.0 sin efecto sistemático en la
discriminación (gap ~0.5–0.7, a veces peor). La pregunta: ¿puede el kernel
*actuar* para lograr objetivos, no solo reaccionar?

## El recorrido de diagnóstico (cuatro capas)

El diagnóstico fue cambiando a medida que se medía. El orden importa porque cada
capa parecía la causa final y no lo era:

1. **Superficial — proyección random.** `_latent_to_action` es un MLP de pesos
   fijos no entrenables. Medido: la acción casi no dependía del latente real
   (L2≈0.006 vs latente=0), porque la salida estaba dominada por los *biases
   constantes* del MLP, no por el latente. El latente, sin embargo, **sí es
   informativo**: `pred(latente real)` vs `pred(latente=0)` = 0.47 (el predictor
   del world model lo usa con fuerza).

2. **Media — falta de exploración.** Un agente reactivo nunca toma acciones
   diversas, así que el world model nunca ve el efecto de acciones alternativas y
   `imagine()` extrapola basura. Necesario: exploración durante el aprendizaje.

3. **El "muro" (falso) — métrica engañosa.** Con exploración, una medición de
   fidelidad de dinámica dio 2/4 con predicciones casi uniformes (0.24–0.26):
   parecía que el world model **no aprendía** la dinámica acción→consecuencia.
   **Era la métrica.** Se miraba el `argmax` *absoluto* de la predicción, pero la
   observación absoluta está dominada por el `state` lento y autocorrelado. El
   efecto de la acción vive en el **delta** (`Δ ≈ r·(action − state)`). Midiendo
   el delta, la fidelidad es **4/4** ya con 400 steps. El world model **sí
   aprende** la dinámica.

4. **La causa real — desalineamiento de `last_action`.** Un planner EFE sobre el
   world model del kernel *fallaba* (0.638 vs 0.693 reactivo) pese a un world
   model fiel. Causa: `step()` entrenaba el world model con la acción **interna**
   del kernel (`softmax(stimulus)`), no con la acción **ejecutada** por el
   planner. El world model aprendía la dinámica de una acción que no se aplicaba.
   Al alinear (`last_action` = acción ejecutada), el mismo planner pasó a
   **0.971 vs 0.693**.

## La solución

Hacer la **selección de acción dentro de `step()`**, de modo que la acción
elegida sea `actual_self` y fluya a `last_action` — el alineamiento sale gratis,
sin tocar el world model.

`action_mode="efe"` elige el candidato que minimiza la free energy esperada:

    G(a) = KL(preference ‖ softmax(imagine([a])))          [valor pragmático]
           − efe_epistemic_weight · H(softmax(imagine([a])))   [valor epistémico]

`argmin_a G(a)`, con probabilidad `explore_eps` de acción exploratoria. La
receta completa de agencia, validada:

1. World model fiel a la dinámica de acción (ya lo era tras la auditoría).
2. **Exploración** durante el warm-up (para que el modelo vea acciones diversas).
3. Selección por **EFE** sobre candidatos (no `softmax(stimulus)`).
4. **Preferencia C** que define qué se persigue.

Ref: Friston et al. 2015, *Active inference and epistemic value*.

## Impacto medido

Entorno reactivo `state_{t+1}=(1−r)·state+r·action`, objetivo `C=[0.7,0.1,0.1,0.1]`,
900 steps (400 de exploración), promedio de 5 seeds.

| Métrica | Reactivo | Agencia (EFE) |
|---|---|---|
| cosine(state, C) | 0.6934 | **0.9707** (+0.28) |
| Fidelidad dinámica de acción (delta) | — | **4/4** |
| Fidelidad medida en `argmax` absoluto (engañosa) | — | 2/4 (azar) |
| Efecto de `latent_weight` en discriminación Ψ | sin efecto sistemático | — |

## Decisiones de diseño

- **Modo opt-in.** `action_mode="reactive"` (default) es **byte-idéntico** al
  kernel previo (probado por regresión); `latent_weight` se conserva.
- **Preferencia C externa** (param del constructor, normalizada a distribución).
  La semántica de qué persigue el agente se decide por experimento; el MVP da el
  mecanismo.
- **Candidatos por defecto**: las acciones puras (one-hot) + la uniforme — base
  mínima "empujar un canal" vs "quedarse plano". Configurable.
- **Algebraica**: `softmax(obs+Δ) = softmax(imagine([a]))` — el delta fue clave
  para *entender* el bug, pero la fórmula final es directa.

## Extensiones investigadas — y por qué NO se integraron (todavía)

Se prototiparon las tres extensiones (en `yvyra-zeta/fase_epistemico*.py`,
`fase_detour.py`, `fase_wm_benchmark.py`). Hallazgos:

- **Valor epistémico real** (ensemble disagreement, no el proxy de entropía):
  mejora el aprendizaje del world model **+82%** (explorar lo incierto), pero en
  una tarea simple aporta **0%** — el agente pragmático 1-step ya converge
  (0.97). Aporta al *modelo del mundo*, no a tareas que el pragmático resuelve.
- **Horizonte de planning > 1**: no se pudo demostrar valor — el entorno reactivo
  simple no lo necesita (greedy 1-step ya basta).
- **La tijera (hallazgo central)**: las extensiones solo lucen donde el world
  model NO alcanza; donde alcanza, no se necesitan. Conclusión inicial: el cuello
  de botella sería la capacidad del world model.
- **El muro era metodológico, no de capacidad**: un entorno "detour" (dinámica
  condicional discontinua) parecía mostrar que el modelo no aprende (error ~0.51
  estancado con más capacidad/steps). **Era un confound train/eval**: se entrenaba
  con acciones continuas y el planner evaluaba one-hots fuera de distribución,
  disparando un fallback degenerado del entorno. Con un benchmark **limpio**
  (mismo espacio de acción train/eval, eval in-distribution), todas las
  arquitecturas aprenden la dinámica condicional a **~0.04**. Y **más capacidad
  cruda perjudica**: MLP profundo (3×128) fue el peor (0.078) vs MLP base (0.040)
  y MLP-delta (0.037, mejor).

**Decisión (YAGNI):** no integrar epistémico ni horizonte ni un world model más
grande ahora. El MVP pragmático basta para el espacio 4-D actual, y la evidencia
dice que el modelo ya es capaz aquí. Las dos cosas reales pendientes:
1. Que el planner use candidatos de acción consistentes con el entrenamiento (no
   one-hots OOD) — fix del planner, barato.
2. La complejidad real no está en la dinámica 4-D sino en el **mapeo experiencia
   → stimulus** (alta dimensión, secuencias, observabilidad parcial). Un world
   model más capaz (recurrente/probabilístico) se justifica **después** de definir
   ese mapeo, que es lo que fija sus requisitos. Antes es especular.

## Reproducir

```bash
PYTHONPATH=src python -m pytest -q tests/test_agency_efe.py          # 7 tests
PYTHONIOENCODING=utf-8 PYTHONPATH=src python experiments/kernel/exp_agency.py  # PASS 0.97 vs 0.69
```
