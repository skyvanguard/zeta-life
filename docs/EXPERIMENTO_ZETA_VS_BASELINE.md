# Experimento: Zeta vs Baseline Consciousness

## Pregunta de Investigación

**¿Qué aportan los ceros de Riemann a la dinámica de consciencia artificial?**

Hipótesis inicial: La modulación por ceros de Riemann mantiene al sistema en el "borde del caos", produciendo:
- Mayor variabilidad en respuestas
- Transiciones arquetípicas más fluidas
- Individuación más rápida pero estable
- Sueños más compensatorios

---

## Diseño Experimental

### Condiciones Comparadas

| Condición | Descripción | Frecuencias |
|-----------|-------------|-------------|
| **ZETA** | Ceros de Riemann | 14.13, 21.02, 25.01, 30.42, ... |
| **UNIFORM** | Frecuencias uniformes | 14.0, 17.64, 21.28, 24.93, ... |
| **NONE** | Sin modulación | - |
| **RANDOM** | Ruido aleatorio | Gaussiano N(0, 0.3) |

### Protocolo

- **Réplicas**: N=10 por condición
- **Estímulos**: 50-500 por réplica
- **Sueños**: 60-100 pasos post-estímulos
- **Seeds**: Diferentes para cada réplica (42, 142, 242, ...)

### Métricas

**Discretas** (basadas en arquetipo dominante):
- `archetype_entropy`: Entropía de Shannon de distribución de dominantes
- `transition_rate`: Cambios de dominante / total pasos
- `final_integration`: Entropía normalizada del estado final

**Continuas** (capturan diferencias sutiles):
- `trajectory_length`: Suma de distancias entre estados consecutivos
- `position_variance`: Varianza de posiciones entre células
- `blend_entropy`: Entropía promedio de mezcla arquetipal
- `position_oscillation`: Desviación estándar de cambios de posición

---

## Resultados

### Experimento 1: Estímulos Claros (50 pasos)

**Configuración**: Atracción=0.1, estímulos con dirección definida

| Métrica | ZETA | UNIFORM | NONE | RANDOM |
|---------|------|---------|------|--------|
| trajectory_length | 0.398 | 0.398 | 0.328 | 0.346 |
| position_variance | 0.046 | 0.046 | 0.034 | 0.034 |

**Significancia**: ZETA vs NONE (p=0.0003), ZETA vs RANDOM (p=0.0036)

### Experimento 2: Estímulos Ambiguos (50 pasos)

**Configuración**: Atracción=0.1, estímulos cercanos a [0.25, 0.25, 0.25, 0.25]

| Métrica | ZETA | UNIFORM | NONE | RANDOM |
|---------|------|---------|------|--------|
| trajectory_length | 0.390 | 0.390 | 0.318 | 0.336 |
| position_oscillation | 0.002 | 0.002 | 0.002 | 0.002 |

**Significancia**: ZETA vs NONE (p=0.0006), ZETA vs RANDOM (p=0.0036)
Nueva ventaja: position_oscillation vs RANDOM (p=0.0376)

### Experimento 3: Largo Plazo (500 pasos)

**Configuración**: Atracción=0.1, 500 estímulos ambiguos + 100 sueños

| Métrica | ZETA | UNIFORM | NONE | RANDOM |
|---------|------|---------|------|--------|
| trajectory_length | **1.933** | **1.933** | 1.566 | 1.677 |
| position_variance | 0.046 | 0.046 | 0.035 | 0.035 |
| position_oscillation | 0.002 | 0.002 | 0.001 | 0.001 |

**Significancia**:
- ZETA vs NONE: +0.367 (+23%), p=0.0002 ***
- ZETA vs RANDOM: +0.256 (+15%), p=0.0003 ***
- **ZETA vs UNIFORM: Δ≈0, p=1.0** (sin diferencia)

---

## Análisis de Modulación

Se verificó que los moduladores producen valores diferentes:

```
t     | ZETA        | UNIFORM     | Diferencia
------+-------------+-------------+-----------
1     | -0.314      | -0.308      | 0.006
50    | +0.111      | +0.506      | 0.395
100   | -0.618      | +0.140      | 0.759

Correlación ZETA-UNIFORM: 0.80
```

A pesar de producir secuencias diferentes, ambos generan el mismo comportamiento estadístico en el sistema.

---

## Conclusiones

### Hallazgos Principales

1. **La modulación oscilante mejora la dinámica** (p < 0.001)
   - Trayectorias +23% más largas que sin modulación
   - Mayor exploración del espacio arquetipal

2. **Modulación estructurada > ruido aleatorio** (p < 0.001)
   - Trayectorias +15% más largas que con ruido
   - Oscilaciones más regulares

3. **Frecuencias ZETA ≈ Frecuencias UNIFORMES** (p = 1.0)
   - Sin diferencias significativas en ninguna métrica
   - Incluso en experimentos largos (500 pasos)

### Interpretación Teórica

Los ceros de Riemann aportan **dinamismo** al sistema, pero no una ventaja computacional única. El "borde del caos" puede lograrse con cualquier modulación periódica estructurada.

**Lo que importa**:
- Tener ALGUNA modulación (vs ninguna)
- Que sea estructurada (vs aleatoria)

**Lo que NO importa** (en este dominio):
- Las frecuencias específicas (zeta vs uniforme)
- La distribución cuasi-aleatoria de los ceros

### Implicaciones

1. Para aplicaciones prácticas, frecuencias uniformes son igual de efectivas
2. El valor de zeta puede estar en otros dominios (resonancia temporal, memoria, etc.)
3. La elegancia matemática de zeta no se traduce automáticamente en ventaja computacional

---

## Archivos Generados

```
exp_zeta_vs_baseline.py              # Código del experimento
zeta_vs_baseline_20251229_1110.png   # Gráficos exp. 1
zeta_vs_baseline_20251229_1122.png   # Gráficos exp. 2 (métricas continuas)
zeta_vs_baseline_20251229_1127.png   # Gráficos exp. 3 (ambiguos)
zeta_vs_baseline_20251229_1143.png   # Gráficos exp. 4 (500 pasos)
zeta_vs_baseline_results_*.json      # Datos crudos
```

---

## Trabajo Futuro

Posibles experimentos para encontrar dominios donde zeta SÍ aporte ventaja:

1. **Tareas de memoria temporal**: ¿Las frecuencias zeta mejoran recall?
2. **Resonancia con datos estructurados**: ¿Datos con periodicidad zeta?
3. **Aprendizaje**: ¿Diferencias en convergencia de entrenamiento?
4. **Perturbaciones**: ¿Mejor recuperación ante ruido?

---

*Experimento realizado: 2025-12-29*
*Sistema: ZetaConsciousness v1.0*
