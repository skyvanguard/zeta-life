# Índice de Experimentos

Este documento lista todos los experimentos disponibles en el proyecto Zeta-Life/IPUESA.

## Resumen

| Categoría | Scripts | Descripción |
|-----------|---------|-------------|
| Consciousness | 25 | IPUESA y validación jerárquica |
| Organism | 21 | ZetaOrganism y comportamiento emergente |
| Psyche | 9 | ZetaPsyche y arquetipos Jungianos |
| Cellular | 4 | Zeta Game of Life y LSTM |
| Validation | 14 | Validación teórica de kernels zeta |

---

## Consciousness (25 scripts)

### IPUESA Core

| Script | Propósito | Runtime | Paper |
|--------|-----------|---------|-------|
| `exp_ipuesa.py` | Experimento base con condiciones de control | ~5 min | Sección 4 |
| `exp_ipuesa_synth.py` | SYNTH-v1 (primera síntesis) | ~5 min | Sección 4.3 |
| `exp_ipuesa_synth_v2.py` | SYNTH-v2 (configuración final) | ~5 min | Sección 5 |
| `exp_ipuesa_synth_v2_consolidation.py` | Validación estadística N=20 | ~30 min | Sección 6 |
| `exp_ipuesa_scale.py` | Test de escala (50, 100 agentes) | ~20 min | - |

### IPUESA Variantes (Falsificación)

| Script | Hipótesis | Resultado | Paper |
|--------|-----------|-----------|-------|
| `exp_ipuesa_td.py` | Temporal Discount previene riesgo | TSI = -0.517 (invierte) | Tabla 2 |
| `exp_ipuesa_ce.py` | Co-evolución espontánea | MA = 0.0 (falla) | Tabla 2 |
| `exp_ipuesa_sc.py` | Self-Continuity | Parcial | - |
| `exp_ipuesa_ap.py` | Anticipatory Preservation | Parcial | - |
| `exp_ipuesa_rl.py` | Reflexive Loop | Parcial | - |
| `exp_ipuesa_ct.py` | Continuity Token | Parcial | - |
| `exp_ipuesa_ei.py` | Existential Irreversibility | Parcial | - |
| `exp_ipuesa_mi.py` | Meta-Identity | Parcial | - |
| `exp_ipuesa_ae.py` | Adaptive Emergence | Parcial | - |
| `exp_ipuesa_x.py` | Exploratory Expansion | Parcial | - |
| `exp_ipuesa_sh.py` | Hierarchical Self | Parcial | - |

### IPUESA Holográfico

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_ipuesa_hg.py` | Embeddings holográficos básicos | ~5 min |
| `exp_ipuesa_hg_plus.py` | Stress testing de embeddings | ~10 min |
| `exp_ipuesa_hg_cal.py` | Calibración Goldilocks (grid search) | ~15 min |

### Consciousness Jerárquica

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_hierarchical_validation.py` | Validación multi-nivel | ~5 min |
| `exp_validacion_5_mejoras.py` | Validación de 5 mejoras arquitectónicas | ~10 min |

---

## Organism (21 scripts)

### Core

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_organism.py` | Comportamiento básico Fi-Mi | ~2 min |
| `exp_regeneration.py` | Regeneración después de daño | ~3 min |
| `train_organism.py` | Entrenamiento de pesos | ~10 min |

### Multi-Organismo

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_dos_organismos.py` | Interacción 2 organismos | ~5 min |
| `exp_dos_organismos_v2.py` | Versión mejorada | ~5 min |
| `exp_tres_organismos.py` | Interacción 3 organismos | ~5 min |
| `exp_tres_organismos_v2.py` | Versión mejorada | ~5 min |
| `exp_multi_organismo_simple.py` | Multi-organismo básico | ~5 min |
| `exp_multi_organismo_grande.py` | Multi-organismo grande | ~10 min |

### Ecosistemas

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_ecosistema.py` | Ecosistema multi-especie | ~10 min |
| `exp_ecosistema_depredacion.py` | Ecosistema con depredadores | ~10 min |
| `exp_depredacion.py` | Dinámica depredador-presa | ~5 min |
| `exp_lotka_volterra.py` | Validación Lotka-Volterra | ~5 min |
| `exp_simbiosis.py` | Relaciones simbióticas | ~5 min |

### Comportamiento Avanzado

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_comunicacion_quimica.py` | Comunicación química | ~5 min |
| `exp_migracion.py` | Comportamiento migratorio | ~5 min |
| `exp_migracion_v2.py` | Migración mejorada | ~5 min |
| `exp_escasez_energia.py` | Respuesta a escasez | ~5 min |
| `exp_estres_masivo.py` | Estrés masivo | ~10 min |
| `exp_escalabilidad.py` | Tests de escala | ~15 min |
| `exp_escenarios_avanzados.py` | Escenarios complejos | ~10 min |

### LSTM Comparaciones

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_memoria_temporal.py` | Memoria temporal zeta | ~5 min |
| `exp_organism_lstm_comparison.py` | Comparación con LSTM | ~10 min |
| `exp_organism_lstm_hard.py` | LSTM bajo estrés | ~10 min |

---

## Psyche (9 scripts)

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_anima_emergente.py` | Emergencia del arquetipo Anima | ~5 min |
| `exp_anima_compensacion.py` | Compensación entre arquetipos | ~5 min |
| `exp_porque_anima.py` | Análisis del rol del Anima | ~5 min |
| `exp_self_reflection.py` | Ciclos de auto-reflexión | ~5 min |
| `exp_decay_vs_nodecay.py` | Efecto del decay en estabilidad | ~10 min |
| `exp_zeta_vs_baseline.py` | Comparación zeta vs uniforme | ~15 min |
| `experimento_estabilidad_self.py` | Estabilidad del Self | ~5 min |
| `experimento_self_realizado.py` | Individuación completa | ~10 min |
| `experimento_suenos.py` | Consolidación de sueños | ~5 min |

---

## Cellular (4 scripts)

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_learnable_phi.py` | Parámetros φ aprendibles | ~10 min |
| `exp_zeta_gated.py` | Gates con kernel zeta | ~10 min |
| `exp_zeta_lstm_memory_test.py` | Test de memoria LSTM-zeta | ~5 min |
| `exp_zeta_lstm_validation.py` | Validación completa LSTM | ~15 min |

---

## Validation (14 scripts)

### Teoría Zeta

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_teoria_zeta.py` | Validación teórica básica | ~5 min |
| `exp_teoria_zeta_v2.py` | Validación extendida | ~10 min |
| `exp_teoria_zeta_v3.py` | Validación completa | ~15 min |

### Métricas Dinámicas

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_correlation_dimension.py` | Dimensión de correlación | ~10 min |
| `exp_entropy_validation.py` | Validación de entropía | ~10 min |
| `exp_lyapunov_validation.py` | Exponentes de Lyapunov | ~10 min |
| `exp_lyapunov_full.py` | Análisis Lyapunov completo | ~20 min |
| `exp_power_spectrum.py` | Espectro de potencia | ~10 min |

### Resonancia

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_resonance_mini.py` | Resonancia rápida | ~2 min |
| `exp_resonance_fast.py` | Resonancia moderada | ~5 min |
| `exp_resonance_full.py` | Resonancia completa | ~15 min |
| `exp_resonance_comparison.py` | Comparación de resonancias | ~10 min |

### Robustez

| Script | Propósito | Runtime |
|--------|-----------|---------|
| `exp_robust_fast.py` | Test de robustez rápido | ~5 min |
| `exp_robust_comparison.py` | Comparación de robustez | ~10 min |
| `exp_real_validation.py` | Validación con datos reales | ~15 min |

---

## Cómo Ejecutar

```bash
# Ejecutar un experimento específico
python experiments/consciousness/exp_ipuesa_synth_v2.py

# Ejecutar todos los experimentos de una categoría
for f in experiments/consciousness/exp_ipuesa*.py; do python "$f"; done
```

## Requisitos de Sistema

| Tipo | RAM | CPU | GPU |
|------|-----|-----|-----|
| Rápido (<5 min) | 4 GB | 2 cores | No |
| Moderado (5-15 min) | 8 GB | 4 cores | Opcional |
| Largo (>15 min) | 16 GB | 4+ cores | Recomendado |

## Notas

- Todos los experimentos usan `seed=42` para reproducibilidad
- Los resultados se guardan en `results/`
- Las figuras se guardan como PNG en `results/` o timestamped en `results/ipuesa/figures/`
