# Trabajo relacionado y fuentes de inspiración

Escaneo de literatura (junio 2026) mapeado a cada componente del **Conscious
Kernel**. Para cada área: qué papers son relevantes y qué ofrecen — **valida**
(confirma nuestro diseño), **inspira** (a adoptar), **SOTA/gap** (lo que nos
falta), **caución** (riesgo conocido). Honesto: en varios casos la literatura
muestra que vamos por un camino correcto pero con una implementación más simple
que el estado del arte.

---

## 1. Inferencia activa + world model + EFE (el corazón del kernel)

- **The Free Energy Principle for Perception and Action: A Deep Learning Perspective** (arXiv:[2207.06415](https://arxiv.org/abs/2207.06415)). Reseña canónica de inferencia activa con deep learning: world model generativo, planificación por **minimización de la energía libre esperada (EFE)** combinando valor pragmático (objetivo) y epistémico (incertidumbre).
- **Learning Perception and Planning with Deep Active Inference** (arXiv:[2001.11841](https://arxiv.org/abs/2001.11841)); **Deep Active Inference for Delayed/Long-Horizon Environments** (arXiv:[2505.19867](https://arxiv.org/abs/2505.19867)).

→ **VALIDA** que el kernel *es* un agente de inferencia activa estándar (EFE = pragmático − epistémico es exactamente nuestro `_efe_cost`). **GAP:** estos trabajos retropropagan gradientes de EFE en una política y planifican a horizonte largo; nuestro planner es *shooting*/CEM de 1 paso. Ver §2.

## 2. World models / model-based RL — el SOTA contra el que medirnos

- **Dream to Control: Learning Behaviors by Latent Imagination** (DreamerV1, arXiv:[1912.01603](https://arxiv.org/abs/1912.01603)); **Mastering Diverse Domains through World Models** (DreamerV3, arXiv:[2301.04104](https://arxiv.org/abs/2301.04104); versión Nature 2025). Aprende un world model latente, **imagina** trayectorias y entrena un actor-crítico propagando **gradientes de valor** por la dinámica latente. Un único config supera a métodos especializados en 150+ tareas; primer algoritmo en minar diamantes en Minecraft sin datos humanos.

→ **INSPIRA (upgrade #1):** reemplazar nuestra selección de acción por *shooting/CEM* (que mostramos que no escala — `exp_cem.py`) por **imaginación latente + actor-crítico con gradientes de valor** estilo Dreamer. Es el camino directo para que el lado de **control** (donde ya mostramos ventaja, §3.7) compita en tareas reales. **SOTA/gap:** nuestro Mackey-Glass (§3.6) y control viven en 4-D; Dreamer es la referencia externa a la que apuntar.

## 3. Curiosidad por disagreement — exactamente lo que implementamos

- **Planning to Explore via Self-Supervised World Models** (Plan2Explore, Sekar et al., ICML 2020, arXiv:[2005.05960](https://arxiv.org/abs/2005.05960)): novedad = **desacuerdo de un ensemble** de modelos one-step en el espacio latente; planifica hacia la novedad *futura* esperada. SOTA zero-shot en 20 tareas de control.
- **Self-Supervised Exploration via Disagreement** (Pathak et al., ICML 2019).
- **Large-Scale Study of Curiosity** + el problema del **Noisy-TV** (Burda et al., 2019; arXiv:[2102.04399](https://arxiv.org/abs/2102.04399)).

→ **VALIDA la idea:** nuestra señal de disagreement de ensemble (`wm_disagreement_heads`) es literalmente Plan2Explore, y el disagreement es robusto al Noisy-TV donde la curiosidad por error-de-predicción falla. **PERO** matiza nuestro negativo honesto (§3.5): Plan2Explore *funciona* a escala porque usa un ensemble de **dinámica** (no cabezas de latente compartido), planificación profunda (Dreamer) y tareas visuales ricas. Nuestro null se debe al **régimen** (toy 4-D, cabezas shallow, shooting), no a la idea. **INSPIRA:** ensemble de dinámica real + planificación profunda podrían revivir el efecto.

## 4. Memoria complementaria (CLS) — fast/slow

- **Why there are Complementary Learning Systems...** (McClelland, McNaughton & O'Reilly, 1995) y su actualización **What Learning Systems do Intelligent Agents Need? CLS Theory Updated** (Kumaran, Hassabis & McClelland, TICS 2016).
- Implementaciones ML: **AHA — Artificial Hippocampal Algorithm** (arXiv:[1909.10340](https://arxiv.org/abs/1909.10340)); **CLS para continual learning con pattern separation/completion** (arXiv:[2507.11393](https://arxiv.org/abs/2507.11393)).

→ **VALIDA** nuestro `FastMemory` (episódico, gated por sorpresa) + `SlowMemory` (lr lento) + consolidación en el sueño — es CLS fiel. **INSPIRA:** agregar **pattern separation** y *replay* secuencial (no solo orden por sorpresa) para acercarlo al hipocampo real y a continual learning.

## 5. Medir "consciencia" — por qué Ψ es heurística, y qué adoptar

- **Críticas a IIT/Φ:** *The Problem with Phi* (Cerullo, PLOS Comp Biol 2015); *El Φ no está bien definido para sistemas físicos generales* (arXiv:[1902.04321](https://arxiv.org/abs/1902.04321)); calcular Φ es **computacionalmente prohibitivo** y solo factible en sistemas diminutos; carta de 124 firmantes (2023) pidiendo no llamarlo ciencia hasta que sea testeable. Alternativas prácticas: Φ* y heurísticas aproximadas.
- **Consciousness in Artificial Intelligence: Insights from the Science of Consciousness** (Butlin, Long et al., 2023, arXiv:[2308.08708](https://arxiv.org/abs/2308.08708)) y su versión TICS 2025: traduce teorías (RPT, **GWT**, HOT, **Predictive Processing**, **Attention Schema**) en **indicator properties** computacionalmente chequeables; concluye que ningún sistema actual es consciente, pero da un método riguroso por *credencias*.

→ **VALIDA** nuestra decisión de degradar Ψ a heurística (computar Φ real es intratable/mal definido). **INSPIRA (reframe importante):** en vez de un Ψ bespoke, evaluar el kernel contra los **indicator properties** de Butlin — y el kernel ya tiene varios: *recurrencia* (GRU latente), *global workspace* (`ConsciousOrganism`), *modelo predictivo* (world model), *precisión/atención* (precisiones aprendidas ≈ attention schema), *agencia* (EFE). Ese es el lenguaje honesto y publicable para hablar de "consciencia" aquí, no Ψ.

## 6. LLM-agents + inferencia activa + auto-reporte — directamente para Yvyra

- **Active Inference for Self-Organizing Multi-LLM Systems** (Prakki, 2024, arXiv:[2412.10425](https://arxiv.org/abs/2412.10425)): una **capa cognitiva de inferencia activa por encima** de un LLM ajusta prompts/estrategias minimizando energía libre — exactamente la topología del puente Yvyra (kernel como capa de control sobre un agente LLM).
- **Active inference para prompting confiable en medicina** (npj Digital Medicine 2025): actor-crítico (Therapist/Supervisor) — patrón de "sugerencia" como el nuestro.
- **Caución (el riesgo de Yvyra):** *LLMs Report Subjective Experience Under Self-Referential Processing* (arXiv:[2510.24797](https://arxiv.org/abs/2510.24797)) y *Evidence for Limited Metacognition in LLMs* (arXiv:[2509.21545](https://arxiv.org/abs/2509.21545)): el auto-reporte de un LLM existe pero es **limitado e inconsistente** — justo el riesgo que `YVYRA_BRIDGE.md` señala ("si Yvyra puntúa arbitrariamente, el kernel integra ruido").

→ **INSPIRA + VALIDA** la arquitectura Yvyra (capa de inferencia activa sobre LLM) y **advierte:** hay que validar la consistencia de los 4 ejes auto-reportados antes de confiar en Ψ sobre ellos.

---

## Qué adoptar, en orden de valor

1. ✅ **HECHO — Imaginación latente + actor-crítico estilo Dreamer** (`action_mode="dreamer"`, `exp_dreamer.py`): el actor amortizado iguala/supera a la búsqueda en control model-based a ~60–130× menos costo por acción, con ventaja creciente en dimensión. Falta llevarlo a un benchmark RL externo (item 4).
2. **Adoptar el marco de indicator properties de Butlin** para el claim de "consciencia" — mapear el kernel a recurrencia/GWT/predictive/attention/agency en vez de (o además de) Ψ. Bajo costo, alta integridad y publicable. [§5]
3. **Ensemble de dinámica real** (no cabezas de latente compartido) + planificación profunda para reintentar la curiosidad por disagreement en un régimen donde Plan2Explore sí funciona. [§3]
4. 🟡 **PARCIAL — Benchmark externo de control** (CartPole-v1, `exp_cartpole.py`): el kernel transfiere parcialmente (greedy 166 vs random 22; 33% del óptimo) como regulación de inferencia activa, pero el actor-crítico online es inestable. Falta estabilizarlo (replay) y probar DMC/continuo.
5. **Validar el auto-reporte de Yvyra** (consistencia de los 4 ejes) antes de cerrar el loop, grounded en la literatura de metacognición LLM. [§6]

> Lectura honesta del escaneo: el kernel está **alineado con líneas de investigación legítimas** (inferencia activa, world models, Plan2Explore, CLS, indicator properties), pero con implementaciones **más simples** que el SOTA. El mayor retorno está en adoptar la maquinaria Dreamer (control) y el marco de Butlin (consciencia), no en más adornos.
