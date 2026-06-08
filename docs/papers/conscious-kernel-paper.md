# El Conscious Kernel: integración coherente emergente por inferencia activa

**Autor:** Francisco Ruiz
**Fecha:** Junio 2026
**Estado:** reescritura del proyecto alrededor de su centro de gravedad real. Reemplaza, como tesis vigente, al paper *"Zeta-Life: Un Framework Unificado…"* (Enero 2026), que se conserva como registro histórico con su erratum.

---

## Resumen

Presentamos el **Conscious Kernel**: una unidad adaptativa de **inferencia activa** (active inference) para IA, que implementa un ciclo
`PERCIBIR → PREDECIR → COMPARAR → ACTUALIZAR → MEMORIZAR → ACTUAR → REFLEXIONAR → SOÑAR`
sobre un modelo del mundo aprendido, un modelo de sí mismo recursivo, errores de predicción ponderados por precisión, memoria complementaria (rápida/lenta), selección de acción por energía libre esperada (EFE), e identidad persistente entre sesiones. Sobre el kernel se construye un **organismo darwiniano** multi-kernel.

El proyecto nació integrando los ceros de la función zeta de Riemann con sistemas de vida artificial. Esta reescritura documenta un cambio honesto de tesis: **los valores específicos de zeta no son load-bearing** fuera de un dominio (autómatas celulares espaciales); el motor real es la inferencia activa. Reportamos, con el mismo rigor, lo que **funciona** y lo que **no**: (i) un índice de integración Ψ **auto-calibrante** y robusto; (ii) que una red equiespaciada (Fourier) **iguala o supera** a zeta en el camino temporal; (iii) **control continuo** que alcanza objetivos arbitrarios (cierra el verbo "controlar"); (iv) que el refinamiento CEM **no aporta** en control unimodal; y (v) que una señal epistémica genuina (**disagreement** de ensemble) **sí impulsa exploración/curiosidad** donde un proxy de entropía no lo hacía. El código, 16 experimentos y 475 tests son reproducibles.

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
Dinámica latente aprendida: `encoder (Linear→ReLU→Linear)` + `GRUCell` (transición) + `predictor (Linear)`. Aprende online de error de predicción (prior) y de un paso posterior de reconstrucción que entrena el encoder. `imagine()` hace rollouts contrafácticos sin mutar el estado. **Ensemble opcional de cabezas** (`disagreement_heads`) con optimizer separado y *bootstrap masking*: su varianza (disagreement) es alta donde el modelo no aprendió → señal epistémica (§3.5).

### 2.2 Modelo de sí mismo (`self_model.py`)
Embedding de identidad persistente, actualizado por EMA, más una vía de auto-predicción entrenada por gradiente (canal interoceptivo). La "reflexión" itera un GRU a profundidad fija; hay una línea de auto-referencia (`embed + self_embedding`), pero la presentamos como recurrencia con un toque auto-referencial, **no** como un "Strange Loop" en sentido fuerte.

### 2.3 Errores precision-weighted y energía libre (`prediction_error.py`)
Energía libre reportada = término de *accuracy* ponderado por precisión, `F = Σ_i precisión_i · ||error_i||²`. Lo más principista es el **aprendizaje de precisiones**: optimizan el objetivo completo (con el término `−log precisión`), convergiendo a la varianza-inversa del error — que es lo que una precisión de inferencia activa *es*. No es una energía libre variacional completa (no hay término de complejidad/KL ni densidad posterior explícita): lo declaramos como **predictive-coding con precisiones aprendidas**.

### 2.4 Memoria complementaria y sueño (`complementary_memory.py`, `dream_engine.py`)
**CLS** real: `FastMemory` (búfer episódico, gated por sorpresa) + `SlowMemory` (red semántica con lr lento). El `DreamEngine` consolida rápido→lento y reproduce identidad; su *ritmo* de fases usa el kernel zeta `K_σ(t)` (el único uso "load-bearing" de zeta en el kernel, y es solo scheduling — §3.2).

### 2.5 Selección de acción (EFE) (`conscious_kernel.py`)
- `reactive`: `acción = softmax(estímulo)`.
- `efe`: minimiza energía libre esperada `G(a) = KL(C ‖ norm(imagine(a))) − w·epistémico`. Soporta **candidatos continuos** (muestreo en el símplex, consistente con el entrenamiento), **horizonte** (rollout sostenido), **CEM** (refinamiento por cross-entropy) y normalización **L1 fiel** de la observación.
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

### 3.5 Curiosidad por disagreement: un positivo (`exp_curiosity.py`)
Dotando al término epistémico de una señal **real** (disagreement de ensemble) en un entorno de dos regímenes que esconde dinámica tras la exploración, el agente **curioso visita el régimen novel ~2× más** que el pragmático (0.30 vs 0.13 de fracción de tiempo, 8 semillas). La dirección es reproducible; la varianza es alta (la exploración lo es). Es la **primera extensión que ayuda** — el disagreement genuino logra lo que el proxy de entropía no.

---

## 4. Discusión

### 4.1 Ψ es una heurística, no una medida de consciencia
Ψ no es la φ de IIT ni una energía libre; es un parámetro de orden acotado, calibrado para discriminar coherencia de ruido. Lo declaramos como tal y documentamos sus constantes como **calibración**. Su valor está en ser una señal monótona y robusta de integración, útil (p. ej.) como material introspectivo para el agente acoplado (Yvyra).

### 4.2 Zeta, degradado a componente probado
El nombre histórico permanece, pero la evidencia propia desmonta su rol central. Lo honesto es presentar zeta como una **decisión de diseño probada y mayormente falsada** (un resultado negativo valioso), salvo en CA espacial.

### 4.3 El patrón "tijera"
Recurrentemente, las extensiones (horizonte, epistémico-proxy, CEM, espectro especial) **lucen solo donde el método simple falla**; donde el método simple basta, no se necesitan. La curiosidad por disagreement (§3.5) es la excepción que ayuda porque ataca un fallo real (exploración) que el método simple no resuelve.

### 4.4 Ledger de honestidad (qué ayudó y qué no)

| Extensión | ¿Ayudó? | Evidencia |
|---|---|---|
| Ψ auto-calibrante (sin clamp) | **Sí** | brecha plana 0.68 vs colapso del clamp fijo |
| Frecuencias zeta (temporal) | **No** | Fourier iguala/supera; GUE plano |
| Acción continua EFE (control) | **Sí** | error 0.05 vs 0.35 (one-hots) |
| Horizonte > 1 | No | sin ganancia en control inercial |
| CEM | No (confiable) | dentro del ruido, signo cambia |
| Curiosidad por disagreement | **Sí** | 2× exploración del régimen novel |

### 4.5 Limitaciones
- Energía libre = solo término de accuracy (sin KL/complejidad).
- El ensemble es de cabezas con latente compartido (captura incertidumbre del predictor, no de la transición).
- Entornos de baja dimensión (4-D símplex); falta validación en tareas ricas/reales (objetivo del puente Yvyra).
- Ψ sigue dependiendo de una constante de escala FE→Φ (`psi_fe_scale`), documentada como calibración, no derivada.

---

## 5. Trabajo relacionado
Inferencia activa / FEP (Friston et al.); Complementary Learning Systems (McClelland, O'Reilly et al.); exploración por desacuerdo de modelos (p. ej. Plan2Explore, *disagreement-based exploration*); Integrated Information Theory (Tononi) como vocabulario, no como implementación; Global Workspace Theory (Baars) como metáfora de arbitraje. La conexión número-teórica original (Montgomery–Odlyzko sobre la estadística GUE de los ceros) se documenta como hipótesis testeada, no usada de forma load-bearing.

## 6. Conclusión
El Conscious Kernel es un sustrato de inferencia activa coherente y honesto que **aprende** (predicción), **controla** (EFE continuo), **integra** (Ψ robusto), **explora con curiosidad** (disagreement) y tiene un camino para **acoplarse a un agente vivo** (Yvyra). El aporte metodológico es tanto el sistema como la disciplina de falsación: incluyendo el desmontaje del propio claim que dio nombre al proyecto.

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
PYTHONPATH=src python experiments/kernel/exp_yvyra_bridge.py
```
Salidas (figuras + `.txt`) en `results/`. Diseño del puente en `docs/YVYRA_BRIDGE.md`; índice de integración en `src/zeta_life/integration/formal_equations.py`.
