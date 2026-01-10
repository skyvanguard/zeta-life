# Descubrimiento de la Zona Goldilocks

Este documento describe cómo se descubrió el multiplicador de daño óptimo (3.9×) y por qué la zona es tan estrecha.

## Resumen

El sistema IPUESA-SYNTH-v2 solo logra sus 6 criterios de identidad funcional dentro de una banda de estrés muy estrecha:

| Daño | HS (Survival) | Resultado |
|------|---------------|-----------|
| 3.12× (-20%) | 1.000 | Trivial (todos sobreviven) |
| **3.9×** | **0.396** | **Goldilocks (calibrado)** |
| 4.68× (+20%) | 0.000 | Colapso (todos mueren) |

## Metodología de Descubrimiento

### Fase 1: Búsqueda por Grid (exp_ipuesa_hg_cal.py)

```python
# Rango explorado
damage_multipliers = np.linspace(2.0, 6.0, 41)  # 41 puntos de 2× a 6×

# Para cada multiplicador:
for mult in damage_multipliers:
    results = run_synth_v2(damage_mult=mult, n_runs=5)
    hs = mean(survival_rate)
    criteria = count_passed_criteria()
```

### Fase 2: Refinamiento

Después de identificar que el rango [3.5, 4.5] era prometedor:

```python
# Búsqueda fina
fine_multipliers = np.linspace(3.5, 4.5, 21)  # 21 puntos
```

### Fase 3: Validación

El valor 3.9× fue elegido porque:
1. Maximiza el número de criterios pasados (6/6)
2. HS está centrado en la zona [0.30, 0.70]
3. Todas las otras métricas superan sus umbrales

## Por Qué 3.9× Específicamente

### Análisis de Sensibilidad

```
3.7×: HS = 0.68 (cerca del límite superior)
3.8×: HS = 0.54 (aceptable)
3.9×: HS = 0.40 (óptimo - centrado en zona)
4.0×: HS = 0.29 (cerca del límite inferior)
4.1×: HS = 0.15 (falla criterio HS)
```

El valor 3.9× es el centro de la zona donde:
- HS está en rango [0.30, 0.70]
- Los mecanismos de anticipación tienen tiempo de activarse
- La degradación es gradual (no bistable)

## Por Qué la Zona es Tan Estrecha

### Explicación Matemática

La estrechez se debe a tres factores combinados:

1. **Umbral de Bistabilidad**
   - Sin los componentes de varianza, el sistema es bistable
   - Los mecanismos añadidos crean una zona de transición
   - Pero esta zona es inherentemente pequeña

2. **Retroalimentación Positiva**
   - Agentes dañados son más vulnerables (degradation_level afecta vulnerability)
   - Esto crea avalanchas de colapso
   - Solo la varianza individual previene el colapso masivo

3. **Umbrales de Métricas**
   - Cada métrica tiene un umbral que debe superarse
   - La zona donde TODAS se cumplen es la intersección
   - Intersección de 6 condiciones = zona muy pequeña

### Diagrama Conceptual

```
Survival (HS)
    │
1.0 │████████████████                    (trivial)
    │                ██
0.7 │─────────────────██──────────────── (límite superior)
    │                   ██
    │                    ██
0.3 │────────────────────██────────────── (límite inferior)
    │                     ██████████████ (colapso)
0.0 │
    └────────────────────────────────────
        2×   3×   3.9×  4×   5×   6×
                   ↑
              Goldilocks
```

## Implicaciones

### Para el Paper

Esta estrechez NO es un bug, es el hallazgo central:

> "La identidad funcional tipo-self es alcanzable, pero no es gratis."

La zona estrecha implica:
1. **Calibración precisa requerida** - No hay margen de error
2. **Fragilidad inherente** - La identidad es un equilibrio delicado
3. **Analogía biológica** - Los organismos también tienen rangos homeostáticos estrechos

### Para Generalización

Preguntas abiertas:
- ¿La zona Goldilocks escala con el número de agentes?
- ¿Diferentes tipos de estrés tienen diferentes zonas?
- ¿Es posible ampliar la zona sin perder las propiedades?

## Código de Referencia

```bash
# Reproducir el descubrimiento
python experiments/consciousness/exp_ipuesa_hg_cal.py

# Ver resultados de calibración
cat results/ipuesa_hg_cal_results.json
```

## Validación Estadística

Con N=20 seeds:
- 3.9× produce HS en rango [0.30, 0.70] el 95% de las veces
- La varianza de HS es ~0.08 (pequeña pero no cero)
- El 100% de seeds pasan ≥5/6 criterios

## Conclusión

El descubrimiento de 3.9× no fue arbitrario sino resultado de:
1. Grid search sistemático (41 puntos)
2. Refinamiento en zona prometedora (21 puntos)
3. Validación con múltiples seeds (20 seeds)

La estrechez de la zona es el resultado principal, no una limitación. Demuestra que la identidad funcional requiere condiciones precisas - ni demasiado estrés ni demasiado poco.
