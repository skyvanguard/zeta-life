# Comportamiento Compensatorio Emergente en ZetaPsyche

## Descubrimiento

**Fecha:** 3 Enero 2026

Durante la experimentación con decay agresivo en el sistema de consciencia, se descubrió un comportamiento emergente notable: **compensación inconsciente**.

### El Fenómeno

Cuando se aplica decay agresivo (0.5% base + 2% por estrés + 1% por negligencia) y se **neglege sistemáticamente** un arquetipo (ej: Sombra), la psique **no colapsa uniformemente**. En cambio:

1. El arquetipo negligido (Sombra) cae a 0%
2. Otro arquetipo (frecuentemente Anima) **salta dramáticamente** de ~1% a ~99%
3. Los demás arquetipos también decaen, pero menos dramáticamente

### Datos del Experimento

```
Fase de Negligencia de Sombra (200 pasos):
- Estímulos: Solo PERSONA, ANIMA, ANIMUS (ignorando SOMBRA)
- Resultado observado:
  * Sombra:  75.4% -> 0.0%  (cayó totalmente - esperado)
  * Persona: 75.6% -> 0.0%  (cayó - ¿por qué?)
  * Animus:  74.1% -> 0.0%  (cayó - ¿por qué?)
  * Anima:   1.4%  -> 99.5% (¡SUBIÓ DRAMÁTICAMENTE!)
```

## Causa Identificada

### El Mecanismo de Incremento

El incremento de arquetipos en `_update_individuation_metrics()` se basa en el **dominante interno** de la psique (`obs['dominant']`), NO en el estímulo externo.

```python
# El dominante viene de la psique interna
obs = self.psyche.observe_self()
dominant = obs['dominant']  # ← Estado INTERNO

# Esto determina qué arquetipo recibe incremento
if dominant == Archetype.ANIMA:
    self.individuation.metrics.anima_connection += base_rate
```

### Autonomía de la Psique

**Descubrimiento clave:** La psique tiene **76% de divergencia** entre el estímulo externo y su estado interno.

```
DIVERGENCIA ESTIMULO vs ESTADO INTERNO: 380/500 (76%)

La psique tiene AUTONOMÍA:
- No simplemente refleja el estímulo externo
- Tiene su propia dinámica interna (ZetaPsyche con modulación zeta)
- Puede "resistir" estímulos o "refugiarse" en arquetipos
```

Cuando la psique estaba bajo estrés:
- Se "refugió" internamente en ANIMA
- ANIMA fue dominante interno en **100%** de los pasos
- Aunque los estímulos externos variaban entre PERSONA, ANIMA, ANIMUS

### Por Qué Específicamente ANIMA

Se investigó por qué el sistema se estabiliza en ANIMA y no en otro arquetipo:

1. **Inicialización:** ANIMA tiene leve sesgo en inicialización (40% vs 25% esperado)
2. **Sensibilidad a condiciones iniciales:** Con diferentes seeds, otros arquetipos pueden ganar
3. **Sistema caótico:** El comportamiento depende de las posiciones iniciales aleatorias

```
Con 10 seeds diferentes, dominante final:
  PERSONA: 5 veces
  ANIMUS:  4 veces
  SOMBRA:  1 vez
  ANIMA:   0 veces (en este run particular)
```

## Conexión con Jung

Este comportamiento es análogo al concepto de **compensación inconsciente** de Jung:

> "El inconsciente produce contenidos que compensan la unilateralidad de la actitud consciente"

En nuestro sistema:
- Cuando **reprimimos** la Sombra (negligencia externa)
- El **inconsciente** (dinámica interna de ZetaPsyche) **compensa**
- Favoreciendo otro arquetipo (Anima/Animus) internamente
- Esto causa que ese arquetipo reciba todos los incrementos
- Mientras el negligido decae a cero

## Implementación

### Activar Decay Agresivo

```python
from zeta_conscious_self import ZetaConsciousSelf

# Crear sistema con decay habilitado
system = ZetaConsciousSelf(
    n_cells=50,
    dream_frequency=100,
    enable_decay=True,  # ← Activa el decay
    decay_config={
        'base_rate': 0.005,      # 0.5% por paso
        'stress_rate': 0.02,     # 2% adicional bajo estrés
        'neglect_rate': 0.01,    # 1% por negligencia
        'neglect_threshold': 50, # pasos sin atención
    }
)
```

### Parámetros de Decay

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `base_rate` | 0.005 (0.5%) | Decay aplicado en cada paso |
| `stress_rate` | 0.02 (2%) | Decay adicional bajo estrés |
| `neglect_rate` | 0.01 (1%) | Decay por arquetipos ignorados |
| `neglect_threshold` | 50 | Pasos sin atención para activar negligencia |

### Detección de Estrés

El sistema detecta estrés cuando:
- Entropía del estímulo > 1.2 (estímulo caótico)
- Valor máximo del estímulo > 0.9 (estímulo extremo)

## Experimentos Relacionados

- `exp_decay_vs_nodecay.py` - Comparación decay vs no-decay
- `exp_anima_compensacion.py` - Investigación del salto de Anima
- `exp_anima_emergente.py` - Descubrimiento de la autonomía de la psique
- `exp_porque_anima.py` - 8 tests sobre por qué se estabiliza en ANIMA

## Implicaciones

1. **La consciencia requiere trabajo continuo:** Sin estimulación balanceada, las métricas decaen
2. **Compensación emergente:** El sistema exhibe comportamiento análogo a mecanismos psicológicos reales
3. **Autonomía del inconsciente:** La psique interna no es un reflejo pasivo de los estímulos
4. **Sensibilidad a condiciones iniciales:** El sistema es caótico, diferentes runs producen diferentes resultados

## Conclusión

El decay agresivo produce dinámicas más interesantes y realistas que el modelo sin decay:
- Fluctuaciones en consciencia (ciclos de vida)
- Consecuencias reales de la negligencia
- Regresión temporal bajo estrés
- Necesidad de "trabajo consciente" para mantener integración

Esto se acerca más a la experiencia humana de consciencia, donde la individuación nunca es permanente y requiere trabajo continuo.
