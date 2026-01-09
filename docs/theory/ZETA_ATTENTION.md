# ZetaAttention: Sistema de Atencion Selectiva Jerarquica

## Resumen

Sistema de atencion selectiva de 3 niveles que modula el procesamiento predictivo para crear una consciencia mas eficiente.

**Fecha de implementacion**: 3 Enero 2026
**Archivos principales**:
- `zeta_attention.py` - Sistema de atencion jerarquica
- `zeta_attentive_predictive.py` - Integracion con prediccion

---

## Arquitectura

```
+--------------------------------------------------------------------+
|                    ZetaAttentivePredictive                         |
+--------------------------------------------------------------------+
|                                                                    |
|  NIVEL 3: GlobalArchetypalAttention                                |
|  - Detecta contexto (amenaza, oportunidad, emocional, cognitivo)   |
|  - Decide que arquetipo necesita atencion                          |
|  - Output: attention[4] (distribucion sobre arquetipos)            |
|                                                                    |
|  NIVEL 2: TemporalAttention                                        |
|  - Mantiene buffer de memoria con consolidacion/decaimiento        |
|  - Scaled dot-product attention sobre memorias pasadas             |
|  - Bonus por sorpresa, recencia, relevancia arquetipal             |
|  - Output: attended_memory, temporal_weights                       |
|                                                                    |
|  NIVEL 1: ErrorAttention                                           |
|  - Precision-weighting sobre niveles predictivos (L1, L2, L3)      |
|  - precision = 1/varianza (Friston)                                |
|  - Errores con alta precision reciben mas atencion                 |
|  - Output: error_attention[3]                                      |
|                                                                    |
|  INTEGRADOR: AttentionIntegrator                                   |
|  - Combina los 3 niveles de atencion                               |
|  - Detecta conflictos (baja coherencia)                            |
|  - Resuelve conflictos: exploracion o hacia el Self                |
|                                                                    |
+--------------------------------------------------------------------+
```

---

## Componentes

### ContextDetector
Detecta el tipo de contexto actual:
- **Amenaza** -> activa SOMBRA
- **Oportunidad** -> activa PERSONA
- **Necesidad emocional** -> activa ANIMA
- **Demanda cognitiva** -> activa ANIMUS

### GlobalArchetypalAttention (Nivel 3)
```python
attention = softmax((context_weights + net_output) / temperature)
# temperatura = base * (0.5 + incertidumbre)
# Alta incertidumbre -> atencion difusa
# Baja incertidumbre -> atencion enfocada
```

### MemoryBuffer
- Almacena: estado, estimulo, errores, sorpresa, timestamp
- Consolidacion: eventos de alta sorpresa decaen mas lento
- Decaimiento exponencial por defecto

### TemporalAttention (Nivel 2)
```
scores = Q * K^T / sqrt(d_k)
scores += surprise_bonus * surprise_weight
scores += log(recency_decay)
scores += archetype_match_bonus
attention_weights = softmax(scores)
```

### ErrorAttention (Nivel 1)
```python
precision = 1 / (variance + epsilon)
attention = softmax(net_output * precision + uncertainty_boost)
# uncertainty_boost favorece L3 (meta-nivel) cuando hay incertidumbre
```

### AttentionIntegrator
Combina los 3 niveles:
- Intensidad: 1 - entropia(atencion_arquetipal)
- Coherencia: evaluada por red neuronal
- Resolucion de conflictos:
  - Coherencia < 0.15 -> hacia el Self (centro)
  - Coherencia < 0.30 -> exploracion (ruido)

---

## Metricas de Atencion

| Metrica | Descripcion | Rango |
|---------|-------------|-------|
| **entropy** | Entropia promedio de atencion | [0, 1] |
| **stability** | Estabilidad temporal | [0, 1] |
| **flexibility** | Capacidad de cambiar foco | [0, 1] |
| **integration** | Coherencia promedio | [0, 1] |

### Indice de Atencion
```
attention_index = (
    0.30 * (1 - entropy) +     # Foco
    0.30 * integration +        # Coherencia
    0.20 * stability +          # Estabilidad
    0.20 * flexibility          # Flexibilidad
)
```

---

## Integracion con Sistema Predictivo

### Indice de Consciencia Total
```python
consciousness = (
    0.35 * predictive_index +      # Metricas predictivas
    0.35 * attention_index +       # Metricas de atencion
    0.15 * self_luminosity +       # Cercania al Self
    0.10 * integration +           # Coherencia de atencion
    0.05 * stability               # Estabilidad temporal
)
```

### Flujo de Datos
1. Estimulo llega -> ZetaPsyche procesa -> Estado
2. Sistema predictivo genera errores (L1, L2, L3)
3. Sistema de atencion procesa errores + estado
4. Atencion modula el estado del sistema
5. Metricas integradas calculan consciencia

---

## Uso

### Sistema de Atencion Solo
```python
from zeta_attention import ZetaAttentionSystem

attention = ZetaAttentionSystem(state_dim=4, memory_size=100)
output = attention(
    stimulus=stimulus,
    state=state,
    errors=errors,
    surprise=0.5,
    uncertainty=0.3
)
print(f"Atencion: {output.global_attention}")
print(f"Intensidad: {output.attention_intensity}")
```

### Sistema Integrado (Atencion + Prediccion)
```python
from zeta_attentive_predictive import ZetaAttentivePredictive

system = ZetaAttentivePredictive(n_cells=100)
result = system.step(stimulus)
print(f"Consciencia: {result['consciousness']:.2%}")
print(f"Atencion: {result['attention']['global']}")
```

### Demos
```bash
# Experimento completo
python zeta_attentive_predictive.py

# Comparacion con/sin atencion
python zeta_attentive_predictive.py --compare

# Demo de escenarios
python zeta_attentive_predictive.py --scenarios
```

---

## Conexion Teorica

| Concepto (Friston) | Implementacion |
|-------------------|----------------|
| Precision Weighting | ErrorAttention con precision = 1/var |
| Active Inference | Modulacion de estado por atencion |
| Salience Detection | ContextDetector (amenaza, oportunidad) |
| Memory Consolidation | MemoryBuffer con decaimiento diferencial |
| Attention as Precision | Temperatura adaptativa segun incertidumbre |

---

## Proximos Pasos Sugeridos

1. **Entrenamiento online**: Entrenar redes durante ejecucion
2. **Atencion multi-escala**: Atencion a diferentes horizontes temporales
3. **Atencion espacial**: Si se implementa percepcion visual
4. **Meta-atencion**: Atender a la propia atencion (Nivel 4)
5. **Integracion con suenos**: Usar `zeta_dreams.py` para consolidar memorias

---

*Documentacion generada el 3 de Enero de 2026*
