# Cómo Reproducir los Resultados del Paper IPUESA

Este documento mapea cada figura y tabla del paper a los scripts que las generan.

## Requisitos

```bash
# Instalar el proyecto
pip install -e ".[full]"

# Verificar instalación
python -c "from zeta_life.core import MemoryAwarePsyche; print('OK')"
```

## Regenerar Todas las Figuras del Paper

```bash
# Genera las 7 figuras del paper en docs/papers/figures/
python scripts/generate_paper_figures.py
```

## Mapeo Figura → Script

| Figura | Descripción | Script | Comando |
|--------|-------------|--------|---------|
| Fig 1 | Arquitectura SYNTH-v2 | `generate_paper_figures.py` | `python scripts/generate_paper_figures.py` |
| Fig 2 | Línea temporal falsificación | `generate_paper_figures.py` | (mismo) |
| Fig 3 | Goldilocks Zone | `generate_paper_figures.py` | (mismo) |
| Fig 4 | Ablation heatmap | `generate_paper_figures.py` | (mismo) |
| Fig 5 | Repeatability distributions | `generate_paper_figures.py` | (mismo) |
| Fig 6 | Radar comparison | `generate_paper_figures.py` | (mismo) |
| Fig 7 | Mecanismos problema→solución | `generate_paper_figures.py` | (mismo) |

## Mapeo Tabla → Experimento

### Tabla 1: Definición Operacional (Sección 3.2)

Los umbrales y métricas provienen de la configuración SYNTH-v2:

```bash
python experiments/consciousness/exp_ipuesa_synth_v2.py
```

**Métricas esperadas:**
| Métrica | Umbral | Valor SYNTH-v2 |
|---------|--------|----------------|
| TAE | > 0.15 | 0.216 |
| EI | > 0.30 | 1.0 |
| MSR | > 0.15 | 0.501 |
| ED | > 0.10 | 0.358 |
| deg_var | > 0.02 | 0.0278 |
| HS | [0.30, 0.70] | 0.391 |

### Tabla 2: Falsificación Progresiva (Sección 4)

| Experimento | Script | Resultado esperado |
|-------------|--------|-------------------|
| IPUESA-TD | `exp_ipuesa_td.py` | TSI = -0.517 (invierte) |
| IPUESA-CE | `exp_ipuesa_ce.py` | MA = 0.0 (sin propagación) |
| SYNTH-v1 | `exp_ipuesa_synth.py` | Bistable (0% o 100%) |

```bash
# Ejecutar TD
python experiments/consciousness/exp_ipuesa_td.py

# Ejecutar CE
python experiments/consciousness/exp_ipuesa_ce.py

# Ejecutar SYNTH v1
python experiments/consciousness/exp_ipuesa_synth.py
```

### Tabla 3: Goldilocks Zone (Sección 5.3)

```bash
python experiments/consciousness/exp_ipuesa_hg_cal.py
```

**Resultado esperado:**
| Damage | HS | Outcome |
|--------|-----|---------|
| 3.12× | 1.000 | Trivial |
| 3.9× | 0.396 | Goldilocks |
| 4.68× | 0.000 | Colapso |

### Tabla 4: Ablation Study (Sección 6.1)

```bash
python experiments/consciousness/exp_ipuesa_synth_v2_consolidation.py
```

Este script ejecuta la configuración completa y las variantes de ablación.

### Tabla 5: Repeatability (Sección 6.3)

El mismo script `exp_ipuesa_synth_v2_consolidation.py` ejecuta 16 seeds.

## Orden de Ejecución Recomendado

Para reproducir el paper completo en orden:

```bash
# 1. Falsificación (experimentos que fallan)
python experiments/consciousness/exp_ipuesa_td.py
python experiments/consciousness/exp_ipuesa_ce.py

# 2. SYNTH v1 (bistable)
python experiments/consciousness/exp_ipuesa_synth.py

# 3. SYNTH v2 (éxito)
python experiments/consciousness/exp_ipuesa_synth_v2.py

# 4. Calibración Goldilocks
python experiments/consciousness/exp_ipuesa_hg_cal.py

# 5. Consolidación final (ablation + repeatability)
python experiments/consciousness/exp_ipuesa_synth_v2_consolidation.py

# 6. Generar figuras
python scripts/generate_paper_figures.py
```

## Archivos de Resultados

Los resultados se guardan en:

```
results/
├── ipuesa_td_results.json      # Temporal Discount
├── ipuesa_ce_results.json      # Co-Evolution
├── ipuesa_synth_results.json   # SYNTH v1
├── ipuesa_synth_v2_results.json        # SYNTH v2
├── ipuesa_synth_v2_consolidation.json  # Ablation + seeds
└── ipuesa/
    ├── figures/                # Figuras timestamped
    └── data/                   # Datos timestamped
```

## Validación de Reproducción

Para verificar que tus resultados coinciden con los del paper:

```python
import json

# Cargar resultados
with open('results/ipuesa_synth_v2_results.json') as f:
    results = json.load(f)

# Verificar métricas
expected = {
    'HS': (0.30, 0.70),      # Goldilocks range
    'TAE': (0.15, 1.0),      # > 0.15
    'MSR': (0.15, 1.0),      # > 0.15
    'ED': (0.10, 1.0),       # > 0.10
    'deg_var': (0.02, 1.0),  # > 0.02
    'EI': (0.30, 1.0),       # > 0.30
}

for metric, (lo, hi) in expected.items():
    val = results['final_metrics'].get(metric, 0)
    status = "PASS" if lo <= val <= hi else "FAIL"
    print(f"{metric}: {val:.3f} [{lo}-{hi}] {status}")
```

## Tiempos de Ejecución Aproximados

| Script | Tiempo | GPU |
|--------|--------|-----|
| exp_ipuesa_td.py | ~2 min | No |
| exp_ipuesa_ce.py | ~2 min | No |
| exp_ipuesa_synth.py | ~5 min | No |
| exp_ipuesa_synth_v2.py | ~5 min | No |
| exp_ipuesa_synth_v2_consolidation.py | ~20 min | No |
| generate_paper_figures.py | ~10 seg | No |

## Troubleshooting

### ImportError

```bash
# Reinstalar el paquete
pip install -e .
```

### Resultados diferentes

Los resultados pueden variar ligeramente debido a:
- Diferente seed (todos los experimentos usan `seed=42`)
- Diferente versión de NumPy/PyTorch

Para reproducción exacta, usar:
```python
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
```

### Memoria insuficiente

Reducir `n_agents` en el script si tienes poca RAM:
```python
config['n_agents'] = 16  # Default: 24
```
