# ZetaOrganism: Organismo Artificial con Inteligencia Colectiva

## Vision General

ZetaOrganism es un sistema multi-agente que simula un organismo artificial donde la inteligencia emerge de la interaccion entre celulas que actuan como fuerzas (Fi) o masas (Mi).

## Modelo Fisico

Basado en la dinamica Fi-Mi:
- **Fi (Fuerza Inicial)**: Atrae y controla masas
- **Mi (Masa)**: Sigue a Fi a traves de gradientes
- **Equilibrio**: Fi escala con `sqrt(masa_controlada)`
- **Corrupcion**: Masas pueden convertirse en Fi competidores

## Algoritmo de Comportamiento

Implementa las formulas de las notas de investigacion:
- `A <-> B`: Interaccion bidireccional
- `A = AAA*A`: Auto-similitud recursiva
- `A^3 + V -> B^3 + A`: Transformacion con potencial vital
- `B = AA* - A*A`: Rol neto (Fi vs Mi)

## Uso

```python
from zeta_organism import ZetaOrganism

# Crear organismo
org = ZetaOrganism(grid_size=64, n_cells=100)
org.initialize(seed_fi=True)

# Simular
for _ in range(200):
    org.step()

# Analizar
metrics = org.get_metrics()
print(f"Fi: {metrics['n_fi']}, Coordinacion: {metrics['coordination']:.3f}")
```

## Metricas de Inteligencia

- **Coordinacion**: Que tan agrupadas estan las masas alrededor de Fi
- **Estabilidad**: Homeostasis del sistema (baja varianza de energia)
- **Emergencia**: Aparicion de nuevos Fi desde masas

## Estructura de Archivos

```
cell_state.py       - Estados y transiciones de celula (MASS/FORCE/CORRUPT)
force_field.py      - Campo de fuerzas con kernel zeta
behavior_engine.py  - Algoritmo A<->B, A^3+V->B^3+A
organism_cell.py    - Celula con NCA + memoria Resonant gateada
zeta_organism.py    - Sistema completo multi-agente
exp_organism.py     - Experimentos de inteligencia colectiva
```

## Experimento

Ejecutar:
```bash
python exp_organism.py
```

Genera `zeta_organism_experiment.png` con:
- Evolucion de roles (Fi/Mass/Corrupt)
- Metricas de inteligencia (coordinacion/estabilidad)
- Estado espacial final
- Evolucion de energia

## Referencia Teorica

- Paper: "IA Adaptativa a traves de la Hipotesis de Riemann"
- Notas de investigacion sobre dinamica Fi-Mi y colapso dimensional
- ZetaLSTM Resonant: "Detectar, no imponer"

## Tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Tests de integracion
python -m pytest tests/test_integration.py -v
```
