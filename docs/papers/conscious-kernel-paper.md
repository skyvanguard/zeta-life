# El Conscious Kernel: integración coherente emergente por inferencia activa

**Autor:** Francisco Ruiz
**Fecha:** Junio 2026
**Estado:** reescritura del proyecto alrededor de su centro de gravedad real. Reemplaza, como tesis vigente, al paper *"Zeta-Life: Un Framework Unificado…"* (Enero 2026), que se conserva como registro histórico con su erratum.

---

## Resumen

Presentamos el **Conscious Kernel**: una unidad adaptativa de **inferencia activa** (active inference) para IA, que implementa un ciclo
`PERCIBIR → PREDECIR → COMPARAR → ACTUALIZAR → MEMORIZAR → ACTUAR → REFLEXIONAR → SOÑAR`
sobre un modelo del mundo aprendido, un modelo de sí mismo recursivo, errores de predicción ponderados por precisión, memoria complementaria (rápida/lenta), selección de acción por energía libre esperada (EFE), e identidad persistente entre sesiones. Sobre el kernel se construye un **organismo darwiniano** multi-kernel.

El proyecto nació integrando los ceros de la función zeta de Riemann con sistemas de vida artificial. Esta reescritura documenta un cambio honesto de tesis: **los valores específicos de zeta no son load-bearing** fuera de un dominio (autómatas celulares espaciales); el motor real es la inferencia activa. Reportamos, con el mismo rigor, lo que **funciona** y lo que **no**: (i) un índice de integración Ψ **auto-calibrante** y robusto; (ii) que una red equiespaciada (Fourier) **iguala o supera** a zeta en el camino temporal; (iii) **control continuo** que alcanza objetivos arbitrarios (cierra el verbo "controlar"); (iv) que el refinamiento CEM **no aporta** en control unimodal; y (v) que una señal epistémica de **disagreement** (de un ensemble de dinámica independiente) **sí impulsa exploración** bajo comparación controlada **cuando el término epistémico está ponderado de forma conmensurable** (pareado +0.30, t=3.7, 9/10) — corrigiendo un null previo que estaba *subponderado*, y un sobre-claim aún anterior que era un artefacto de RNG; un caso de estudio de seguir la evidencia hasta revisar la propia conclusión en ambas direcciones. Un primer **benchmark externo** (Mackey-Glass) es **acotante**: como agente condicionado por acción, el kernel queda por debajo de baselines simples en predicción pura — pero en **control model-based bajo dinámica desconocida** ese mismo diseño es una **ventaja** (alcanza el objetivo donde un controlador model-free fracasa), y un **actor amortizado estilo Dreamer** entrenado en imaginación iguala/supera a la búsqueda a ~60–130× menos costo por acción. En un benchmark RL **externo** (CartPole-v1) el kernel **transfiere parcialmente** (~164 vs 22 de random; ~33% del óptimo) como regulación de inferencia activa pura, sin reward externo; un loop con **replay de transiciones** estilo DreamerV3 mejora la curva pero no cierra la brecha (colapso tardío). Construir un **RSSM de referencia** con paridad DreamerV3 (estado recurrente entrenado sobre secuencias + reward aprendido) **resuelve CartPole al techo (500/500)** — acotando el límite del kernel a su **world-model de un paso**, no a la inferencia activa. La **integración** (`RSSMConsciousKernel`) corre las facultades del kernel —identidad, memoria, sueño, **Ψ**— sobre ese RSSM y **alcanza el techo manteniendo Ψ vivo**. El código, 20 experimentos y 496 tests son reproducibles.

**Palabras clave:** inferencia activa, principio de energía libre, consciencia computacional, integración emergente, curiosidad por disagreement, sistemas darwinianos multi-agente.

---

## 1. Introducción

### 1.1 De "zeta-life" a inferencia activa

El proyecto se llamó *zeta-life* y se justificó sobre la conjetura de que los ceros de Riemann, al vivir en la línea crítica `Re(s)=1/2`, aportan una estructura natural de "borde del caos". Tras experimentos controlados (§3.2), concluimos honestamente que **la aritmética específica de zeta no es la que hace funcionar al sistema**. El centro de gravedad real —y donde se concentra el trabajo reciente y reproducible— es un agente de inferencia activa: el **Conscious Kernel**.

Esta reescritura adopta como tesis lo que el sistema *demuestra*, no lo que su nombre promete:

> Un agente de inferencia activa con modelo del mundo aprendido, precisiones aprendidas, memoria complementaria y selección de acción por energía libre esperada constituye un sustrato coherente y falsable para estudiar **integración emergente** y comportamientos asociables a la cognición (anticipación, agencia, curiosidad, identidad persistente). Los ceros de zeta son una **elección de diseño opcional y probada**, no la tesis.

### 1.2 Ethos: falsación sistemática

Mantenemos la práctica de documentar fracasos con el mismo peso que los éxitos. En este trabajo eso incluye degradar públicamente el claim central del paper anterior (zeta), reportar una extensión que no ayuda (CEM, §3.4) y declarar Ψ como **heurística de ingeniería**, no como medida derivada de consciencia (§4.1).

---

## 2. Arquitectura del Conscious Kernel

Cada `kernel.step(estímulo)` ejecuta un paso del ciclo de inferencia activa. Componentes (`src/zeta_life/kernel/`):

### 2.1 Modelo del mundo (`world_model.py`)
Dinámica latente aprendida: `encoder (Linear→ReLU→Linear)` + `GRUCell` (transición) + `predictor (Linear)`. Aprende online de error de predicción (prior) y de un paso posterior de reconstrucción que entrena el encoder. `imagine()` hace rollouts contrafácticos sin mutar el estado. **Ensemble epistémico opcional**: cabezas de readout sobre el latente compartido (`disagreement_heads`) o —más principista— un **ensemble de dinámica independiente** (`dynamics_ensemble`: cada miembro es su propio MLP `(latente,acción)→obs`), ambos con optimizer separado, *bootstrap masking* y RNG dedicado. La varianza entre miembros (disagreement) es alta donde el modelo no aprendió → señal epistémica para la curiosidad (§3.5).

### 2.2 Modelo de sí mismo (`self_model.py`)
Embedding de identidad persistente, actualizado por EMA, más una vía de auto-predicción entrenada por gradiente (canal interoceptivo). La "reflexión" itera un GRU a profundidad fija; hay una línea de auto-referencia (`embed + self_embedding`), pero la presentamos como recurrencia con un toque auto-referencial, **no** como un "Strange Loop" en sentido fuerte.

### 2.3 Errores precision-weighted y energía libre (`prediction_error.py`)
Energía libre reportada = término de *accuracy* ponderado por precisión, `F = Σ_i precisión_i · ||error_i||²`. Lo más principista es el **aprendizaje de precisiones**: optimizan el objetivo completo (con el término `−log precisión`), convergiendo a la varianza-inversa del error — que es lo que una precisión de inferencia activa *es*. No es una energía libre variacional completa (no hay término de complejidad/KL ni densidad posterior explícita): lo declaramos como **predictive-coding con precisiones aprendidas**.

### 2.4 Memoria complementaria y sueño (`complementary_memory.py`, `dream_engine.py`)
**CLS** real: `FastMemory` (búfer episódico, gated por sorpresa) + `SlowMemory` (red semántica con lr lento). El `DreamEngine` consolida rápido→lento y reproduce identidad; su *ritmo* de fases usa el kernel zeta `K_σ(t)` (el único uso "load-bearing" de zeta en el kernel, y es solo scheduling — §3.2).

### 2.5 Selección de acción (EFE) (`conscious_kernel.py`)
- `reactive`: `acción = softmax(estímulo)`.
- `efe`: minimiza energía libre esperada `G(a) = KL(C ‖ norm(imagine(a))) − w·epistémico`. Soporta **candidatos continuos** (muestreo en el símplex, consistente con el entrenamiento), **horizonte** (rollout sostenido), **CEM** (refinamiento por cross-entropy) y normalización **L1 fiel** de la observación.
- `dreamer`: **actor amortizado** entrenado en imaginación con un crítico y gradientes de valor por la dinámica latente diferenciable (`imagine_grad`); reward = −EFE. Elige acciones a costo O(1), sin búsqueda (§3.8).
- Término epistémico: `entropy` (proxy grueso) o `disagreement` (señal real del ensemble).

### 2.6 Índice de integración Ψ (`integration/formal_equations.py`)
`Ψ = Bⁿ/(Kⁿ+Bⁿ)` (Hill, acotada, default) con umbral crítico `Φ_c = F_i/(α−C)`. **Es una heurística de ingeniería**, no IIT ni una energía libre. El punto-medio de la fuerza de binding `F_i` es **auto-calibrante** (EMA de la precisión media), lo que elimina la constante mágica frágil del diseño anterior (§3.1).

### 2.7 Organismo darwiniano (`conscious_organism.py`)
Múltiples kernels compiten por un `GlobalWorkspace` (winner-take-all); energía conservada y spawn/merge/death generan presión de selección.

### 2.8 Puente Yvyra (`bridge/`)
Aplicación: la experiencia auto-reportada (4 ejes) de un agente LLM vivo se vuelve el *mundo* del kernel; éste devuelve Ψ y una **sugerencia** EFE. Es el camino para que el kernel opere sobre experiencia real, no sintética.

---

## 3. Experimentos y resultados

Todos reproducibles bajo `experiments/kernel/` (semillas fijas). Resumen honesto al final (§4.4).

### 3.1 Ψ discrimina y es robusto
Ψ separa entrada coherente de ruido (coherente > 0.7, ruido < 0.3 a horizonte largo) y **no colapsa** al entrenar las precisiones. El diseño anterior dependía de un *clamp* fijo (`psi_prec_half`) que había que recalibrar cada vez que mejoraba el sustrato. El punto-medio auto-calibrante lo elimina: barriendo `psi_prec_half` en dos décadas, la brecha de discriminación adaptativa es **plana (0.68, spread 0.000)**, mientras la versión fija **colapsa a 0** para valores ≥ 20 (`exp_psi_robustness.py`).

### 3.2 ¿Importan las frecuencias zeta? (la falsación)
- **Predicción temporal in-kernel** (`exp_zeta_vs_baselines.py`): en un mundo construido con frecuencias zeta, una red **Fourier equiespaciada iguala/supera** a zeta (error ≈ 0.037 vs 0.037; ambas ~35% mejor que random); en un mundo neutral todas empatan (~0.37). La "ventaja de zeta" es *basis-matching*, no especialidad.
- **Estadística de espaciamiento** (`exp_spacing_statistics.py`): la repulsión de niveles GUE de los ceros es real, pero **≤ una red rígida** en radio de cobertura y condicionamiento, y **funcionalmente plana** dentro del kernel (~1.8% de spread entre zeta/GUE/Poisson/uniforme).
- **Dónde zeta sí gana:** únicamente en los **autómatas celulares espaciales** (resultado histórico: +134% de supervivencia vs vecindario Moore, supera también a UNIFORM, p<0.001).
- **Recomendación:** base fija → `fourier`/`log_spaced`; adaptativa → `learned`. Zeta queda como componente opcional documentado.

### 3.3 Control continuo: cerrar "controlar" (`exp_control.py`)
El planner original solo elegía entre acciones one-hot, fuera de la distribución de entrenamiento y **incapaces de representar un objetivo no-vértice**. Con candidatos **continuos** + normalización **L1 fiel**, el kernel alcanza un objetivo arbitrario `C=[0.7,0.1,0.1,0.1]`:

| política | error ‖estado−C‖ |
|---|---|
| reactive | 0.52 |
| EFE discreto (one-hots) | 0.35 |
| **EFE continuo** | **0.05** |

El horizonte > 1 **no aportó** aquí (consistente con la investigación de agencia).

### 3.4 CEM: un negativo honesto (`exp_cem.py`)
El Cross-Entropy Method está implementado, pero **no mejora de forma confiable** al random shooting en el control inercial: barriendo dimensión 4→32 a presupuesto igualado, la diferencia cambia de signo y queda dentro del ruido. Razón estructural: la tarea es unimodal (la acción óptima *es* el objetivo), sin paisaje rugoso que explotar. CEM se conserva como capacidad para regímenes más duros.

### 3.5 Curiosidad por disagreement: el arco completo (`exp_curiosity.py`)
Un entorno de dos regímenes esconde dinámica tras la exploración (régimen A = casa, régimen B = dinámica permutada no aprendida). El término epistémico recibe una señal **real** (disagreement de ensemble). El recorrido honesto, en tres etapas:

1. **Sobre-claim → retractación (RNG).** Una versión preliminar mostró al curioso visitando B ~2× más, pero una revisión adversarial halló un **confound de RNG** (construir/entrenar el ensemble corría el stream global del que el sampler EFE toma). Corregido (RNG dedicado + el pragmático carga el mismo ensemble con peso 0).
2. **Null bajo control… pero subponderado.** Con el confound corregido y `efe_epistemic_weight=30`, el efecto era ~0 (pareado +0.025, n.s.). Lo reportamos como negativo. **El error sutil:** el disagreement es de magnitud O(1e-3), así que con peso 30 el término epistémico es despreciable frente al pragmático (KL O(0.1–1)) — estaba **subponderado**, no inerte.
3. **Corrección honesta: con peso conmensurable, funciona.** Al revisar item #3 construimos un **ensemble de dinámica independiente** (cada miembro su propio MLP `(latente,acción)→obs`, no una cabeza-readout sobre un latente compartido; contraste de disagreement 2.18× novel/home). Con `efe_epistemic_weight=500` (conmensurable con el pragmático), la curiosidad **impulsa exploración de forma confiable** (pareado, controlado, 10 semillas):

   | señal | curioso vs pragmático (tiempo en B) | diff pareada |
   |---|---|---|
   | heads (latente compartido) | 0.65 vs 0.32 | **+0.33 (t=4.9, 9/10)** |
   | ensemble independiente | 0.56 vs 0.26 | **+0.30 (t=3.7, 9/10)** |

   **Conclusión actualizada:** el disagreement-curiosity **sí** impulsa exploración cuando el término epistémico está ponderado de forma conmensurable; el null previo fue un artefacto de **peso**, no de la idea. El ensemble independiente es la señal más **principista** (consistente y significativa en todo el barrido de pesos 400–1000; las cabezas son más sensibles al peso), aunque a este peso los tamaños de efecto son comparables. Caso de estudio de seguir la evidencia hasta revisar la **propia** conclusión previa — en ambas direcciones.

### 3.6 Tarea externa: Mackey-Glass (`exp_realtask.py`)
Primer test fuera del símplex auto-construido: predicción one-step de **Mackey-Glass** (τ=17), un benchmark caótico canónico, contra baselines honestos (NMSE sobre la mitad de test):

| modelo | NMSE ↓ |
|---|---|
| AR(16) lineal | ~0.000 |
| persistencia | 0.022 |
| GRU plano | 0.023 |
| kernel WM (núcleo, señal cruda) | 0.046 |
| kernel (loop agente, acción=softmax(obs)) | 0.181 |

Hallazgo honesto, en dos partes: (a) el **núcleo** (world model con señal cruda) *transfiere* pero queda ~2× por debajo de un GRU dedicado; (b) el **loop agente completo** es ~8× peor, porque su transición recibe `softmax(obs)` (constante para obs_dim=1) y se pierde la señal — el kernel es un **agente condicionado por acción, no un predictor de secuencias**, y en predicción pura ese diseño es un handicap real. Además, AR lineal resuelve el horizonte de 1 paso (~0), así que ningún modelo no-lineal es necesario ahí. Es el primer dato de benchmark externo y acota dónde sirve el core.

### 3.7 Control model-based: la ventaja espejo (`exp_modelcontrol.py`)
Si la condición-por-acción es un handicap en predicción, debería ser una **ventaja** en control donde la dinámica hay que aprenderla. Test: control con **dinámica de acción desconocida (permutada)** — `state_{t+1} = norm((1−r)·state + r·P[action])` con `P` una permutación fija oculta — hacia un target no-vértice. Un controlador **model-free** (action = C) está **derrotado por la permutación** (manda C, el estado va a P[C]); solo un agente que aprende `P` y la invierte (action ≈ P⁻¹[C]) llega.

| política | error ‖estado−C‖ |
|---|---|
| reactive | 0.52 |
| naive (model-free, action=C) | 0.85 |
| **kernel EFE (model-based)** | **0.046** |

El kernel **aprende la dinámica permutada y la invierte** para alcanzar el target (18× mejor que el model-free, bajo varianza). Es el complemento honesto de §3.6: **handicapeado en predicción pura, ventajoso en control model-based bajo dinámica desconocida** — un caracterización coherente de un agente de inferencia activa.

### 3.8 Planificación amortizada estilo Dreamer (`exp_dreamer.py`)
La selección de acción por *búsqueda* (shooting/CEM) cuesta O(n_samples) rollouts del world model por acción. La adoptamos al patrón **Dreamer**: un **actor amortizado** entrenado en imaginación (con crítico y **gradientes de valor** por la dinámica latente diferenciable; reward = −EFE). En control de dinámica permutada, barriendo la dimensión:

| D | dreamer (amortizado) | mejor search | costo/acción |
|---|---|---|---|
| 4 | 0.046 | 0.040 | **127× más barato** |
| 8 | **0.081** | 0.114 | **60× más barato** |
| 16 | **0.100** | 0.159 | **131× más barato** |

El actor **iguala** a la búsqueda en D=4 y la **supera** en D=8/16 (la búsqueda con `n_samples` fijo se degrada al crecer la dimensión; el actor generaliza), eligiendo acciones a **costo O(1)** (un forward) — **~60–130× más barato por acción**. Es el upgrade #1 de la lista de trabajo relacionado, y un positivo claro: **mismo o mejor control, inferencia mucho más barata, y la ventaja crece con la dimensión.**

### 3.9 Benchmark RL externo: CartPole-v1 (`exp_cartpole.py`)
Primer test en un entorno RL externo reconocido (gymnasium), enmarcado como **regulación de inferencia activa**: preferencia = estado vertical (goal), reward = −distancia (`dreamer_reward="neg_distance"`), acción 2-símplex → argmax → acción discreta. **Sin reward externo.**

| política | largo de episodio (cap 500) |
|---|---|
| random | 22 |
| heurística (casi óptima) | 500 |
| **kernel (eval greedy)** | **~164 ± 21** |

El kernel **transfiere**: aprende a balancear ~164 pasos (**~7× sobre random, ~33% del techo**) puramente como regulación a un estado-objetivo. Honesto y **acotante**: queda muy por debajo del óptimo.

**Estabilización (el arco honesto):** (1) estabilizadores estilo DreamerV3 (target critic EMA, normalización de returns, grad clip) elevaron el training tail 105→132 pero la curva seguía oscilando con caídas catastróficas. (2) Implementamos el **loop con replay de transiciones** completo (buffer de observaciones re-encodeadas con el modelo actual + grounding del world model en transiciones diversas + imaginación desde estados de replay). Esto **mejoró la forma de la curva** —sube de forma clara hasta un **pico ~250 (50% del techo)** en vez de oscilar alrededor de ~100— **pero NO resolvió el problema**: sobre entrenamiento largo aparece un **colapso tardío** (pico → declive) y el eval greedy se mantiene en ~33% del techo, sin superar a los estabilizadores simples. El replay también mejoró la sanidad del control model-based (`exp_dreamer.py`, D=8: 0.047 vs 0.081). Honesto: el replay de transiciones es la arquitectura más principista y eleva el pico, pero **cerrar la brecha al techo requiere paridad DreamerV3 completa** (world model recurrente entrenado sobre **secuencias**/RSSM + **reward model** aprendido), más allá del replay. Es el primer dato en un benchmark RL externo — **transferencia parcial real, no SOTA**.

### 3.10 Paridad DreamerV3: un RSSM crackea CartPole (`exp_dreamerv3.py`)
Para **acotar** de dónde viene la brecha del kernel, construimos un agente **DreamerV2/V3-style** autocontenido (`kernel/rssm.py`, `kernel/dreamerv3_agent.py`): un **RSSM** (estado recurrente determinista `h` + estocástico `z`, prior/posterior Gaussianos, KL balanceada + free nats) **entrenado sobre secuencias** de un replay flat, con **reward y continue heads aprendidos**, y actor-crítico en imaginación (λ-returns, REINFORCE + entropía, target critic EMA).

| política | largo de episodio (cap 500) |
|---|---|
| random | 22 |
| heurística | 500 |
| kernel (world-model de 1 paso) | ~164 (33%) |
| **RSSM DreamerV3-style (eval greedy)** | **500 (100%)** |

El RSSM **resuelve CartPole al techo** (greedy 500/500) en ~8k pasos. Esto **acota el límite del kernel a su world-model**: no es el diseño de inferencia activa lo que falla en CartPole, sino el modelo de **un paso** sin estado recurrente entrenado sobre secuencias ni reward aprendido.

### 3.11 Integración: las facultades del kernel sobre el RSSM (`exp_rssm_kernel.py`)
Cerramos el lazo: `RSSMConsciousKernel` (`kernel/rssm_kernel.py`) **integra** las facultades de consciencia del kernel —identidad persistente (self-model), memoria complementaria (CLS rápida/lenta), consolidación en el sueño con ritmo zeta, y el índice de integración **Ψ**— **sobre** el world-model RSSM y su controlador. Las facultades corren sobre la feature recurrente `s = [h, z]` **sin tocar el camino de control**, reutilizando las clases del kernel (`SelfModel`, `PredictionErrorEngine`, `FastMemory`/`SlowMemory`, `DreamEngine`, las ecuaciones de Ψ) instanciadas sobre el espacio de features.

Resultado: el **kernel integrado alcanza el techo de CartPole** (greedy ~500/500) —las facultades no rompen el control— y **Ψ se mantiene vivo** como señal de integración sobre el estado recurrente (sube con el aprendizaje). Es la respuesta constructiva a §3.10: el kernel de inferencia activa, **dotado de un world-model recurrente entrenado sobre secuencias**, resuelve la tarea externa manteniendo su ciclo de consciencia. (Es una composición de referencia; fundir el RSSM dentro del `ConsciousKernel` canónico —reemplazando su world-model de un paso in situ— es el siguiente paso de ingeniería.)

---

## 4. Discusión

### 4.1 Ψ es una heurística, no una medida de consciencia
Ψ no es la φ de IIT ni una energía libre; es un parámetro de orden acotado, calibrado para discriminar coherencia de ruido. Lo declaramos como tal y documentamos sus constantes como **calibración**. Su valor está en ser una señal monótona y robusta de integración, útil (p. ej.) como material introspectivo para el agente acoplado (Yvyra).

Para hablar de "consciencia" **con rigor** —en vez de tratar Ψ como una pseudo-medida— auditamos el kernel contra las **indicator properties** de Butlin & Long et al. (2023), el lenguaje de la ciencia de la consciencia. Resultado honesto y conservador (`docs/INDICATOR_PROPERTIES.md`): satisface con fuerza los indicadores de **predictive processing (PP-1)** y **agencia/embodiment (AE-1, AE-2)** —su núcleo de diseño—, de forma **parcial** los de global-workspace, recurrencia y higher-order (8/14), y **carece** de attention-schema (AST-1), quality-space (HOT-4) y querying secuencial del workspace (GWT-4). Crucialmente, y siguiendo a los propios autores: **tener indicadores no es ser consciente** — esto actualiza credencias sobre propiedades funcionales, no afirma experiencia.

### 4.2 Zeta, degradado a componente probado
El nombre histórico permanece, pero la evidencia propia desmonta su rol central. Lo honesto es presentar zeta como una **decisión de diseño probada y mayormente falsada** (un resultado negativo valioso), salvo en CA espacial.

### 4.3 El patrón "tijera"
Recurrentemente, las extensiones (horizonte, epistémico-proxy, CEM, espectro zeta, y —tras corregir el confound— también el disagreement-curiosity) **no aportan de forma confiable** en estos regímenes 4-D. La curiosidad parecía la excepción, pero no sobrevivió a una comparación controlada (§3.5). Lo que sí ayuda es estructural: precisiones aprendidas, acción continua (espacio de acción correcto) y memoria complementaria. La lección meta: separar la extensión del confound (RNG, espacio de acción, basis-matching) suele convertir un "ayuda" en un "no aporta".

### 4.4 Ledger de honestidad (qué ayudó y qué no)

| Extensión | ¿Ayudó? | Evidencia |
|---|---|---|
| Ψ auto-calibrante (sin clamp) | **Sí** | brecha plana 0.68 vs colapso del clamp fijo |
| Frecuencias zeta (temporal) | **No** | Fourier iguala/supera; GUE plano |
| Acción continua EFE (control) | **Sí** | error 0.05 vs 0.35 (one-hots) |
| Horizonte > 1 | No | sin ganancia en control inercial |
| CEM | No (confiable) | dentro del ruido, signo cambia |
| Curiosidad por disagreement | **Sí** (con peso conmensurable) | pareado +0.30, t=3.7, 9/10 a `weight=500`; el null previo (+0.025) estaba subponderado (señal O(1e-3)); el "2×" original sí era artefacto de RNG. Ensemble de dinámica independiente = señal más principista |
| Control model-based (dinámica desconocida) | **Sí** | kernel 0.046 vs naive model-free 0.85 (aprende e invierte la permutación) |
| Planificación amortizada (Dreamer) | **Sí** | iguala/supera a la búsqueda (D=16: 0.100 vs 0.159) a ~60–130× menos costo/acción |
| Benchmark RL externo (CartPole-v1) | **Parcial** | greedy ~164 vs random 22 vs óptimo 500: transfiere (~7×), no SOTA; replay de transiciones mejora la curva (pico ~50% del techo) pero hay colapso tardío |
| Paridad DreamerV3 (RSSM de referencia) | **Sí** | un RSSM entrenado sobre secuencias + reward aprendido **resuelve CartPole (greedy 500/500)** → la brecha del kernel es su world-model de 1 paso, no la inferencia activa |
| Integración: facultades del kernel + RSSM | **Sí** | `RSSMConsciousKernel` alcanza el techo de CartPole (~500/500) con identidad/memoria/sueño/**Ψ vivo** sobre el estado recurrente |

### 4.5 Limitaciones
- Energía libre = solo término de accuracy (sin KL/complejidad).
- Entornos mayormente de baja dimensión. El benchmark externo de *predicción*
  (Mackey-Glass, §3.6) es ACOTANTE (el kernel no es un predictor SOTA). El de
  *control* externo (CartPole-v1, §3.9) muestra **transferencia parcial real**
  (~164 vs random 22, ~33% del óptimo) pero **no SOTA**; falta validación en
  experiencia real (Yvyra).
- **El `ConsciousKernel` canónico** (world-model de un paso) se estanca en ~33% de
  CartPole. Un RSSM de referencia lo resuelve (§3.10) y la **integración**
  (`RSSMConsciousKernel`, §3.11) corre las facultades del kernel —identidad,
  memoria, sueño, Ψ— sobre ese RSSM, alcanzando el techo con Ψ vivo. Pendiente:
  **fundir el RSSM dentro del `ConsciousKernel` canónico** (reemplazar su
  world-model in situ), no solo la composición de referencia.
- Ψ sigue dependiendo de una constante de escala FE→Φ (`psi_fe_scale`), documentada como calibración, no derivada.

---

## 5. Trabajo relacionado
Inferencia activa / FEP (Friston et al.; reseña deep-learning arXiv:2207.06415); world models y control model-based por imaginación latente (**Dreamer**, Hafner et al. — el SOTA externo y nuestra principal vía de mejora del planner); exploración por **desacuerdo de ensemble** (**Plan2Explore**, Sekar et al. ICML 2020; Pathak et al. 2019 — la versión a escala de nuestra curiosidad); Complementary Learning Systems (McClelland/O'Reilly 1995; Kumaran et al. TICS 2016); medición de consciencia: críticas a Φ de IIT (intratable / mal definido) que motivan tratar Ψ como heurística, y el marco de **indicator properties** de Butlin & Long et al. (2023) como el lenguaje riguroso a adoptar (recurrencia/GWT/predictive/attention/agency, varios ya presentes en el kernel); LLM-agents + inferencia activa y auto-reporte (Prakki 2024) — relevante al puente Yvyra, con la caución de la metacognición limitada de los LLM. La conexión número-teórica original (Montgomery–Odlyzko sobre la estadística GUE de los ceros) se documenta como hipótesis testeada, no load-bearing. **Lista de lectura completa, mapeada a cada componente, en [`docs/RELATED_WORK.md`](../RELATED_WORK.md).**

## 6. Conclusión
El Conscious Kernel es un sustrato de inferencia activa coherente y honesto que **aprende** (predicción), **controla** (EFE continuo y un actor amortizado estilo Dreamer que iguala/supera a la búsqueda a costo O(1); con su world-model de un paso transfiere parcialmente a CartPole, y **con un world-model RSSM integrado alcanza el techo manteniendo Ψ vivo**, §3.11), **integra** (Ψ robusto) y tiene un camino para **acoplarse a un agente vivo** (Yvyra). El aporte metodológico es tanto el sistema como la disciplina de falsación: incluyendo el desmontaje del propio claim que dio nombre al proyecto y la **retractación de un resultado propio** (la curiosidad) que una revisión adversarial mostró confundido por RNG.

---

## Reproducibilidad
```bash
pip install -e .            # o PYTHONPATH=src
PYTHONPATH=src pytest tests/ -q                 # 475 tests
PYTHONPATH=src python experiments/kernel/exp_psi_robustness.py
PYTHONPATH=src python experiments/kernel/exp_zeta_vs_baselines.py
PYTHONPATH=src python experiments/kernel/exp_spacing_statistics.py --kernel
PYTHONPATH=src python experiments/kernel/exp_control.py
PYTHONPATH=src python experiments/kernel/exp_cem.py
PYTHONPATH=src python experiments/kernel/exp_curiosity.py
PYTHONPATH=src python experiments/kernel/exp_realtask.py
PYTHONPATH=src python experiments/kernel/exp_modelcontrol.py
PYTHONPATH=src python experiments/kernel/exp_dreamer.py
PYTHONPATH=src python experiments/kernel/exp_cartpole.py    # pip install gymnasium
PYTHONPATH=src python experiments/kernel/exp_yvyra_bridge.py
```
Salidas (figuras + `.txt`) en `results/`. Diseño del puente en `docs/YVYRA_BRIDGE.md`; índice de integración en `src/zeta_life/integration/formal_equations.py`.
