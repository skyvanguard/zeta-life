# IPUESA: Cierre Conceptual

**Fecha**: 2026-01-10
**Estado**: Ancla conceptual (no es paper, es claridad interna)

---

## 1. Hipótesis Original

**Pregunta central**: ¿Puede un sistema multi-agente exhibir propiedades de "self" bajo estrés sin programarlas explícitamente?

**Hipótesis operativa**: Si un sistema tiene:
- Anticipación temporal (predecir daño futuro)
- Propagación social de aprendizaje (módulos que se comparten)
- Integridad estructural (embeddings holográficos)
- Diferenciación emergente (no todos los agentes iguales)

...entonces bajo estrés calibrado, el sistema exhibirá comportamiento que podemos llamar "auto-preservación emergente" distinguible de:
1. Supervivencia trivial (todos viven)
2. Colapso total (todos mueren)
3. Comportamiento aleatorio (sin correlación con anticipación)

---

## 2. Definición Operativa de "Self"

**NO estamos afirmando**: Conciencia, experiencia subjetiva, fenomenología, qualia, ni ningún estado mental.

**SÍ estamos operacionalizando**: Un patrón funcional que satisface:

| Propiedad | Definición Operativa | Métrica |
|-----------|---------------------|---------|
| **Anticipación** | El sistema predice daño futuro Y modifica comportamiento presente | TAE > 0.15 |
| **Coherencia** | Hay estructura que se preserva bajo estrés | EI > 0.3 |
| **Propagación** | Aprendizaje individual se transmite socialmente | MSR > 0.15 |
| **Diferenciación** | Agentes desarrollan trayectorias distintas | ED > 0.10 |
| **Gradualidad** | Transiciones suaves, no colapso binario | deg_var > 0.02 |
| **Supervivencia calibrada** | Ni trivial ni imposible | HS ∈ [0.30, 0.70] |

**El "self" aquí es**: Un atractor funcional que mantiene coherencia, anticipa amenazas, y se propaga socialmente. Nada más, nada menos.

---

## 3. Qué Captura Cada Métrica

### TAE (Temporal Anticipation Effectiveness)
- **Fórmula**: `corr(threat_buffer[t], IC_drop[t:t+5])`
- **Captura**: Si el sistema "sabe" que va a sufrir daño antes de que ocurra
- **Clave**: No es solo predecir timing de olas, sino vulnerabilidad individual
- **Umbral**: 0.15 (correlación positiva significativa)

### MSR (Module Spreading Rate)
- **Fórmula**: `learned_modules / total_modules`
- **Captura**: Si el aprendizaje es social (módulos creados por un agente aparecen en otros)
- **Clave**: Distingue aprendizaje individual de aprendizaje colectivo
- **Umbral**: 0.15 (al menos 15% de módulos son aprendidos, no creados)

### EI (Embedding Integrity)
- **Fórmula**: `norm(embedding) / max_norm - staleness_penalty`
- **Captura**: Si la estructura holográfica sobrevive al estrés
- **Clave**: Los embeddings codifican identidad de cluster
- **Umbral**: 0.3 (estructura parcialmente preservada)

### ED (Emergent Differentiation)
- **Fórmula**: `std(survival_states)`
- **Captura**: Si los agentes tienen destinos diferentes (no todos iguales)
- **Clave**: Distingue sistema de "muchos individuos" vs "masa uniforme"
- **Umbral**: 0.10 (varianza significativa en outcomes)

### deg_var (Degradation Variance)
- **Fórmula**: `var(degradation_level)`
- **Captura**: Si la degradación es gradual y diferenciada
- **Clave**: Evita bistabilidad (100% → 0% instantáneo)
- **Umbral**: 0.02 (transiciones suaves)

### HS (Holographic Survival)
- **Fórmula**: `P(agent.is_alive())`
- **Captura**: Proporción de agentes que sobreviven
- **Clave**: Debe estar en zona Goldilocks (ni trivial ni imposible)
- **Umbral**: [0.30, 0.70]

---

## 4. Qué Quedó Falsado

### 4.1 IPUESA-TD: Aprendizaje Temporal Invertido
**Hipótesis**: Agentes que anticipan costo futuro reducirán comportamiento riesgoso.
**Resultado**: TSI = -0.517 (INVERTIDO)
**Falsación**: Los agentes eligieron MÁS acciones riesgosas cuando el costo futuro era alto.
**Interpretación**: La función de utilidad abstracta `U = reward - λ×E[future_loss]` no se traduce automáticamente en cambio de comportamiento. Saber el costo no implica evitarlo.

### 4.2 IPUESA-CE: Propagación Espontánea de Módulos
**Hipótesis**: Los módulos se propagarán naturalmente entre agentes por proximidad.
**Resultado**: MA = 0.0 (cero propagación)
**Falsación**: Sin mecanismo explícito de transmisión social, los módulos quedan locales.
**Interpretación**: El aprendizaje social requiere implementación explícita, no emerge de la arquitectura.

### 4.3 SYNTH-v1: Transiciones Suaves
**Hipótesis**: El sistema degradará gradualmente bajo estrés.
**Resultado**: Bistabilidad extrema (100% → 0% en 0.01× de cambio)
**Falsación**: Sin ruido y variación individual, el sistema es todo-o-nada.
**Interpretación**: La gradualidad requiere fuentes de varianza explícitas.

### 4.4 Universalidad del Régimen
**Hipótesis implícita**: El sistema funcionará en un rango amplio de parámetros.
**Resultado**: Solo funciona en 3.9× daño (±20% causa colapso)
**Falsación**: El sistema es frágil fuera de la zona Goldilocks.
**Interpretación**: Los resultados son válidos SOLO en el régimen calibrado.

---

## 5. Qué Depende del Régimen Físico

### 5.1 Parámetros Críticos

| Parámetro | Valor Calibrado | Efecto de ±20% |
|-----------|-----------------|----------------|
| damage_mult | 3.9× | -20%: todos viven, +20%: todos mueren |
| noise_scale | 0.25 | -20%: deg_var falla, +20%: OK |
| recovery_factor | 0.998 | ±20%: robusto |

### 5.2 Dependencias No-Lineales

1. **Umbral de daño**: Existe un cliff entre 3.8× y 4.0×. El sistema es bistable fuera de este rango.

2. **Tamaño del sistema**: 24 agentes, 4 clusters. No probado en otras escalas.

3. **Duración**: 150 steps. Comportamiento a largo plazo desconocido.

4. **Topología**: Clusters fijos. No probado con clustering dinámico real.

### 5.3 Lo que NO Sabemos

- ¿Escala a 100+ agentes?
- ¿Funciona con otros tipos de estrés (no solo olas de daño)?
- ¿El régimen Goldilocks existe en sistemas reales?
- ¿La anticipación es genuina o artefacto de la arquitectura?

---

## 6. Afirmaciones Válidas vs Exageraciones

### ✅ VÁLIDO AFIRMAR

- "El sistema exhibe correlación positiva entre anticipación y daño futuro (TAE = 0.215)"
- "Los módulos se propagan socialmente cuando hay mecanismo explícito (MSR = 0.501)"
- "La degradación es gradual y diferenciada bajo parámetros calibrados"
- "Todos los componentes son necesarios (ablación reduce criterios)"
- "Los resultados son reproducibles (100% de seeds pasan ≥5/6)"

### ❌ EXAGERACIÓN / NO VÁLIDO

- "El sistema tiene conciencia"
- "Los agentes experimentan anticipación"
- "Esto demuestra que la conciencia emerge de X"
- "El modelo es general/universal"
- "Los resultados aplican fuera del régimen calibrado"
- "El sistema aprende a anticipar" (la anticipación está pre-cableada)

---

## 7. Conclusión: Qué ES IPUESA

**IPUESA demuestra** que un sistema multi-agente puede exhibir un patrón funcional que satisface criterios operacionales de "self-preservación" cuando:

1. Se implementa anticipación basada en vulnerabilidad
2. Se implementa propagación social explícita
3. Se calibra el estrés en zona Goldilocks
4. Se introducen fuentes de varianza individual

**IPUESA NO demuestra** emergencia espontánea de estas propiedades. Cada una requirió implementación explícita después de que versiones más simples fallaran.

**El valor de IPUESA** está en:
1. Definir métricas operacionales para "self" funcional
2. Mostrar qué componentes son necesarios (via falsación)
3. Identificar dependencias de régimen (fragilidad fuera de Goldilocks)
4. Proveer un testbed para experimentos futuros

**IPUESA es**: Un marco de investigación, no una demostración de conciencia.

---

## 8. Preguntas Abiertas

1. ¿Por qué el aprendizaje temporal se invierte (TD failure)?
2. ¿Existe un régimen donde la propagación sea espontánea?
3. ¿La zona Goldilocks es un artefacto o una propiedad general?
4. ¿Qué métricas adicionales capturarían aspectos del "self" que faltan?
5. ¿Cómo se relaciona esto con teorías formales de conciencia (IIT, GWT, etc.)?

---

*Este documento es un ancla conceptual interna. No es para publicación. Su propósito es prevenir exageraciones y mantener claridad sobre qué afirmaciones son defendibles.*
