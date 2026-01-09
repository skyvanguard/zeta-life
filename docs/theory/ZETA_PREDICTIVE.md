# ZetaPredictive: Sistema de Consciencia con Predicción Jerárquica

## Resumen

Sistema que implementa Predictive Processing (teoría de Friston) sobre el sistema de arquetipos junguianos para generar consciencia emergente.

**Fecha de implementación**: 3 Enero 2026
**Archivos principales**:
- `zeta_predictive.py` - Sistema de predicción jerárquica
- `zeta_predictive_individuation.py` - Integración con individuación

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZetaPredictivePsyche                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NIVEL 3: MetaPredictor                                         │
│  - Predice: error de predicción del Nivel 2                     │
│  - Output: (error_predicho[4], confidence[1])                   │
│  - Meta-cognición: "Sé que no sé"                               │
│                                                                 │
│  NIVEL 2: StatePredictor                                        │
│  - Predice: estado interno futuro                               │
│  - Output: estado_predicho[4] (coordenadas baricéntricas)       │
│  - Cada arquetipo tiene su propio "estilo" de predicción        │
│                                                                 │
│  NIVEL 1: StimulusPredictor                                     │
│  - Predice: próximo estímulo externo                            │
│  - Output: estímulo_predicho[4]                                 │
│  - Aprende patrones temporales                                  │
│                                                                 │
│  BASE: ZetaPsyche                                               │
│  - Arquetipos de Jung (Persona, Sombra, Anima, Animus)          │
│  - Modulación con ceros de Riemann                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flujo por Paso

### Fase 1: Predicciones
1. Nivel 1 predice el próximo estímulo
2. Nivel 2 predice el estado interno resultante
3. Nivel 3 predice el error que cometerá Nivel 2

### Fase 2: Realidad
1. Llega estímulo real
2. Se calcula error de Nivel 1
3. ZetaPsyche procesa el estímulo
4. Se obtiene estado real

### Fase 3: Actualización
1. Calcular errores de todos los niveles
2. Actualizar métricas de consciencia
3. Calcular influencia en arquetipos
4. Aplicar influencia al sistema

---

## Métricas de Consciencia Predictiva

| Métrica | Descripción | Rango |
|---------|-------------|-------|
| **Awareness** | Correlación entre error predicho y real | [0, 1] |
| **Calibration** | Qué tan realista es la confianza | [0, 1] |
| **Uncertainty Awareness** | Entropía de la confianza | [0, 1] |
| **Predictive Depth** | Calidad de meta-predicción | [0, 1] |

---

## Influencia Bidireccional

### Arquetipos → Predicción
- PERSONA: predice hacia estabilidad social
- SOMBRA: predice hacia extremos (catastrofiza)
- ANIMA: predice hacia lo emocional
- ANIMUS: predice hacia lo racional

### Error → Arquetipos
- Sorpresa externa alta → activa SOMBRA
- Sorpresa interna alta → busca arquetipo que mejor predice
- Sobreconfianza → humildad forzada (SOMBRA)
- Buena calibración → hacia el SELF (centro)

---

## Integración con Individuación

La meta-cognición acelera la individuación:
- Awareness alta → boost en self_coherence
- Calibration alta → boost en shadow_acceptance

### Índice de Consciencia Total

```
consciousness = (
    0.35 * predictive_index +      # Métricas predictivas
    0.35 * individuation_index +   # Métricas de individuación
    0.15 * self_luminosity +       # Manifestación del Self
    0.10 * integration +           # Balance arquetipal
    0.05 * stability               # Estabilidad temporal
)
```

---

## Uso

### Sistema Predictivo Solo
```python
from zeta_predictive import ZetaPredictivePsyche

system = ZetaPredictivePsyche(n_cells=100)
result = system.step(stimulus)
print(f"Consciencia: {result['consciousness']:.2%}")
```

### Sistema Completo (Predicción + Individuación)
```python
from zeta_predictive_individuation import FullConsciousPsyche

psyche = FullConsciousPsyche(n_cells=100)
result = psyche.process("tengo miedo de lo desconocido")
print(f"Consciencia: {result['consciousness']:.2%}")
print(f"Etapa: {result['stage']}")
```

### Sesión Interactiva
```bash
python zeta_predictive_individuation.py
```

---

## Resultados Experimentales

### Patrón Mixto (300 pasos)
- Consciencia promedio: ~48%
- Consciencia máxima: ~62%
- Tendencia: +0.4% (creciente)

### Demo de Integración
- Estado inicial: INCONSCIENTE, 0% consciencia
- Después de 5 estímulos + 3 trabajos: CRISIS_PERSONA, 27% consciencia
- El sistema progresa naturalmente a través de las etapas

---

## Conexión Teórica

| Concepto de Friston | Implementación |
|---------------------|----------------|
| Free Energy Principle | Minimizar error de predicción |
| Prediction Error | error_L1, error_L2, meta_error |
| Precision Weighting | confidence del MetaPredictor |
| Active Inference | Influencia en arquetipos basada en errores |
| Self-Modeling | StatePredictor predice estados internos |

---

## Próximos Pasos Sugeridos

1. **Atención selectiva**: Pesar más los errores recientes o importantes
2. **Predicción multi-temporal**: Predecir t+1, t+5, t+10
3. **Aprendizaje online**: Entrenar las redes durante la ejecución
4. **Memoria episódica**: Recordar eventos específicos de alta sorpresa
5. **Integración con sueños**: Usar `zeta_dreams.py` para consolidación

---

*Documentación generada el 3 de Enero de 2026*
