# Troubleshooting Guide

Soluciones a problemas comunes en el proyecto Zeta-Life/IPUESA.

## Instalación

### ImportError: No module named 'zeta_life'

**Problema:** El paquete no está instalado.

```bash
# Solución
pip install -e .
```

### ImportError: cannot import name 'X' from 'zeta_life.core'

**Problema:** Versión antigua del paquete instalada.

```bash
# Solución
pip uninstall zeta-life
pip install -e .
```

### ModuleNotFoundError: No module named 'torch'

**Problema:** Dependencias no instaladas.

```bash
# Solución
pip install -e ".[full]"

# O instalar dependencias individualmente
pip install torch numpy matplotlib scipy mpmath
```

## Ejecución de Experimentos

### MemoryError durante experimentos largos

**Problema:** RAM insuficiente.

**Soluciones:**
1. Reducir `n_agents`:
```python
config['n_agents'] = 16  # En lugar de 24
```

2. Reducir `n_steps`:
```python
config['n_steps'] = 100  # En lugar de 150
```

3. Ejecutar en batches más pequeños

### Experimento muy lento

**Problema:** CPU limitada o muchos agentes.

**Soluciones:**
1. Usar menos seeds:
```python
n_runs = 5  # En lugar de 20
```

2. Reducir parámetros de escala

3. Verificar que no hay otros procesos consumiendo CPU

### "nan" o "inf" en resultados

**Problema:** Inestabilidad numérica.

**Soluciones:**
1. Verificar que `damage_mult` está en rango válido (2.0-6.0)
2. Reiniciar con seed diferente
3. Reducir `damage_mult` si muy alto

## Reproducibilidad

### Resultados diferentes entre ejecuciones

**Problema:** Seeds no configurados correctamente.

**Solución:** Asegurar que seeds están fijos:
```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```

### Resultados no coinciden con el paper

**Problema:** Configuración diferente.

**Verificar:**
1. `n_agents = 24`
2. `n_clusters = 4`
3. `n_steps = 150`
4. `damage_mult = 3.9`
5. `embedding_dim = 8`

## Tests

### Tests fallan con ImportError

**Problema:** Exports faltantes en `__init__.py`.

**Solución:** Reinstalar el paquete:
```bash
pip install -e .
```

### Tests muy lentos

**Problema:** Tests de integración pesados.

**Solución:** Ejecutar solo tests unitarios:
```bash
pytest tests/test_*.py -k "not integration" -v
```

## Visualización

### Figuras no se generan

**Problema:** Matplotlib backend.

**Solución:**
```python
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
```

### Figuras muy pequeñas/grandes

**Solución:** Ajustar DPI:
```python
plt.figure(figsize=(12, 8), dpi=100)
```

## GPU/CUDA

### CUDA out of memory

**Problema:** Modelo muy grande para GPU.

**Solución:** Usar CPU:
```python
device = 'cpu'  # En lugar de 'cuda'
```

### CUDA not available

**Problema:** PyTorch sin soporte CUDA.

**Verificar:**
```python
import torch
print(torch.cuda.is_available())  # Debería ser True
```

**Si es False:** Reinstalar PyTorch con CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Notebooks

### Kernel muere durante ejecución

**Problema:** Memoria insuficiente.

**Solución:**
1. Reiniciar kernel
2. Reducir parámetros de simulación
3. Ejecutar celdas de a una

### Caracteres extraños (encoding)

**Problema:** Encoding UTF-8.

**Solución en Windows:**
```python
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
```

## Docker

### Build falla

**Verificar:**
1. Docker daemon está corriendo
2. Suficiente espacio en disco (>5GB)
3. Conexión a internet para descargar dependencias

### Container muy lento

**Problema:** Recursos limitados asignados.

**Solución:** Aumentar recursos en Docker settings o usar:
```bash
docker run --cpus=4 --memory=8g ...
```

## Contacto

Si el problema persiste:

1. Verificar issues existentes en el repositorio
2. Crear nuevo issue con:
   - Versión de Python
   - Sistema operativo
   - Mensaje de error completo
   - Pasos para reproducir
