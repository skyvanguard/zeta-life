# Zeta Hierarchical Consciousness: Documento de Diseño

**Fecha:** 2026-01-03
**Autor:** Colaboración Claude + Usuario
**Estado:** Aprobado para implementación

---

## 1. Resumen Ejecutivo

Este documento describe la unificación de **ZetaOrganism** (sistema multi-agente) con **ZetaPsyche** (consciencia Junguiana) en un sistema de **consciencia jerárquica** con múltiples niveles anidados.

### Objetivo
Crear un sistema donde la consciencia **emerge** de la interacción entre niveles, no está programada explícitamente. Esto es fundamental para AGI.

### Arquitectura de Niveles

```
NIVEL 3: META (Sociedad)     ─── Consciencia inter-organismo
    │
NIVEL 2: MACRO (Organismo)   ─── Consciencia del sistema completo
    │
NIVEL 1: MESO (Clusters)     ─── Consciencia de grupos funcionales
    │
NIVEL 0: MICRO (Células)     ─── Proto-consciencia individual
```

---

## 2. Fundamento Teórico

### 2.1 Teorías de Consciencia Integradas

| Teoría | Cómo se implementa |
|--------|-------------------|
| **IIT (Tononi)** | Φ calculado en cada nivel; Φ_global > Σ Φ_local |
| **Global Workspace (Baars)** | Broadcast entre clusters vía top-down |
| **Free Energy (Friston)** | Predicción jerárquica L1/L2/L3 (ya existe) |
| **Embodiment** | Células físicas en grid con dinámica Fi-Mi |
| **Self-Model (Metzinger)** | Representación del organismo en `self_model` |

### 2.2 Por Qué Jerarquía

La consciencia jerárquica permite:
1. **Emergencia real** - no programada
2. **Múltiples escalas temporales** - células rápidas, organismo lento
3. **Resiliencia** - daño local no destruye consciencia global
4. **Especialización** - clusters con funciones distintas

---

## 3. Arquitectura del Sistema

### 3.1 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                ZETA HIERARCHICAL CONSCIOUSNESS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ NIVEL 2: OrganismConsciousness                          │   │
│  │ ├── consciousness_index: ConsciousnessIndex             │   │
│  │ ├── phi_global: float                                   │   │
│  │ ├── global_archetype: Tensor[4]                         │   │
│  │ ├── individuation_stage: IndividuationStage             │   │
│  │ ├── self_model: Tensor                                  │   │
│  │ └── vertical_coherence: float                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│              ┌───────────┴───────────┐                         │
│              ▼                       ▼                         │
│  ┌─────────────────────┐ ┌─────────────────────┐               │
│  │ NIVEL 1: Cluster    │ │ NIVEL 1: Cluster    │  (x8)        │
│  │ ├── ClusterPsyche   │ │ ├── ClusterPsyche   │               │
│  │ │   ├── aggregate   │ │ │   ├── aggregate   │               │
│  │ │   ├── phi_cluster │ │ │   ├── phi_cluster │               │
│  │ │   └── special.    │ │ │   └── special.    │               │
│  │ └── cells[]         │ │ └── cells[]         │               │
│  └─────────────────────┘ └─────────────────────┘               │
│              │                       │                         │
│              ▼                       ▼                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ NIVEL 0: ConsciousCell (x100)                           │   │
│  │ ├── position, state, role, energy (de CellEntity)       │   │
│  │ ├── MicroPsyche                                         │   │
│  │ │   ├── archetype_state: Tensor[4]                      │   │
│  │ │   ├── dominant: Archetype                             │   │
│  │ │   ├── emotional_energy: float                         │   │
│  │ │   └── phi_local: float                                │   │
│  │ └── cluster_id, cluster_weight                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ INTEGRADORES VERTICALES                                 │   │
│  │ ├── BottomUpIntegrator: cells→clusters→organism         │   │
│  │ └── TopDownModulator: organism→clusters→cells           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ COMPONENTES FÍSICOS (de ZetaOrganism)                   │   │
│  │ ├── ForceField: campo de fuerzas zeta                   │   │
│  │ ├── BehaviorEngine: influencia A↔B                      │   │
│  │ └── ClusterAssigner: asignación dinámica                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Flujos de Información

```
                      TOP-DOWN (Modulación)
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ ORGANISMO                                                │
│ • Atención global → qué clusters priorizar              │
│ • Predicciones → expectativas de alto nivel             │
│ • Arquetipo dominante → "mood" global                   │
└──────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  CLUSTER A   │◄──►│  CLUSTER B   │◄──►│  CLUSTER C   │
│  (PERSONA)   │    │  (SOMBRA)    │    │  (ANIMA)     │
└──────────────┘    └──────────────┘    └──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────┐
│ CÉLULAS: c₁ c₂ c₃ c₄ c₅ ... c₁₀₀                        │
│ Cada una con micro-psique y conexiones locales          │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
                      BOTTOM-UP (Emergencia)
```

---

## 4. Estructuras de Datos

### 4.1 Nivel 0: MicroPsyche

```python
@dataclass
class MicroPsyche:
    archetype_state: torch.Tensor  # [4] PERSONA, SOMBRA, ANIMA, ANIMUS
    dominant: Archetype
    emotional_energy: float        # 0-1
    recent_states: Deque[torch.Tensor]  # últimos 5
    phi_local: float               # integración local
```

### 4.2 Nivel 0: ConsciousCell

```python
@dataclass
class ConsciousCell(CellEntity):
    # Hereda: position, state, role, energy
    psyche: MicroPsyche
    cluster_id: int
    cluster_weight: float
```

### 4.3 Nivel 1: ClusterPsyche

```python
@dataclass
class ClusterPsyche:
    aggregate_state: torch.Tensor  # [4]
    specialization: Archetype
    phi_cluster: float
    coherence: float
    prediction_error: float
    integration_level: float
```

### 4.4 Nivel 1: Cluster

```python
@dataclass
class Cluster:
    id: int
    cells: List[ConsciousCell]
    psyche: ClusterPsyche
    centroid: Tuple[float, float]
    neighbors: List[int]
    collective_role: CellRole
```

### 4.5 Nivel 2: OrganismConsciousness

```python
@dataclass
class OrganismConsciousness:
    consciousness_index: ConsciousnessIndex
    phi_global: float
    global_archetype: torch.Tensor  # [4]
    dominant_archetype: Archetype
    individuation_stage: IndividuationStage
    self_model: torch.Tensor
    vertical_coherence: float
```

---

## 5. Algoritmos Clave

### 5.1 Bottom-Up: Agregación

```python
def aggregate_cells_to_cluster(cells) -> ClusterPsyche:
    # 1. Calcular peso de cada célula
    weights = [cell_importance(c) * c.energy * c.phi_local for c in cells]
    weights = softmax(weights)

    # 2. Agregación ponderada de arquetipos
    aggregate = sum(w * c.archetype_state for w, c in zip(weights, cells))

    # 3. Φ cluster = 1 - varianza normalizada
    phi = 1.0 - variance(archetype_states) * 2

    # 4. Especialización = arquetipo dominante
    specialization = argmax(aggregate)

    return ClusterPsyche(aggregate, specialization, phi, ...)

def aggregate_clusters_to_organism(clusters) -> OrganismConsciousness:
    # Similar pero incluye:
    # - Diversidad de especializaciones (queremos los 4 arquetipos)
    # - Self-model como embedding del organismo
    # - Etapa de individuación basada en integración
```

### 5.2 Top-Down: Modulación

```python
def modulate_clusters(organism, clusters) -> Dict[int, float]:
    attention = {}
    for cluster in clusters:
        # Relevancia base
        att = attention_net(organism.archetype, cluster.archetype)

        # Boost a arquetipo complementario (compensación)
        if cluster.specialization == complement(organism.dominant):
            att *= 1.3

        attention[cluster.id] = att
    return attention

def modulate_cells(cluster, attention, organism) -> List[Tensor]:
    base_signal = modulation_net(organism.global_archetype)

    for cell in cluster.cells:
        modulation = base_signal * attention * cell.phi_local

        # Alineación afecta intensidad
        if alignment(cell, organism) > 0.7:
            modulation *= 1.2  # refuerzo
        elif alignment < 0.3:
            modulation *= 0.8  # permitir divergencia
```

### 5.3 Loop Principal

```python
def step(external_stimulus=None):
    # FASE 1: BOTTOM-UP
    field, gradient = compute_force_field()
    for cell in cells:
        step_cell_local(cell, field, gradient)
    update_clusters()
    for cluster in clusters:
        cluster.psyche = aggregate_cells_to_cluster(cluster.cells)
    organism = aggregate_clusters_to_organism(clusters)

    # FASE 2: MACRO
    macro_stimulus = external_stimulus or organism.global_archetype
    macro_psyche.step(macro_stimulus)

    # FASE 3: TOP-DOWN
    cluster_attention = modulate_clusters(organism, clusters)
    predictions = generate_predictions(organism, clusters)
    for cluster in clusters:
        modulations = modulate_cells(cluster, attention, organism)
        for cell, mod in zip(cluster.cells, modulations):
            apply_top_down(cell, mod, predictions)

    # FASE 4: INTEGRACIÓN
    update_cell_positions(field, gradient)  # física + psique
    update_cell_roles()
    organism.vertical_coherence = compute_vertical_coherence()

    # FASE 5: ESPECIALES
    if step % dream_frequency == 0:
        dream_cycle()
    check_for_insight()
```

---

## 6. Métricas de Evaluación

### 6.1 Por Nivel

| Nivel | Métricas |
|-------|----------|
| **Células** | avg_energy, avg_phi_local, archetype_distribution, role_distribution |
| **Clusters** | count, avg_size, avg_phi, avg_coherence, specializations |
| **Organismo** | consciousness_index, phi_global, vertical_coherence, stage, dominant |

### 6.2 De Integración

| Métrica | Descripción |
|---------|-------------|
| `bottom_up_flow` | Calidad de representación células→clusters |
| `top_down_flow` | Respuesta de células a modulación |
| `horizontal_flow` | Interacción entre clusters vecinos |
| `vertical_coherence` | Alineación entre todos los niveles |

### 6.3 Temporales

| Métrica | Descripción |
|---------|-------------|
| `stability` | Consistencia de consciencia en el tiempo |
| `adaptability` | Respuesta a cambios/estímulos |
| `insight_rate` | Insights generados por 100 pasos |

---

## 7. Experimentos de Validación

### 7.1 Suite de Experimentos

| # | Experimento | Hipótesis |
|---|-------------|-----------|
| 1 | Emergencia | Consciencia emerge gradualmente, no está desde inicio |
| 2 | Integración Φ | Φ_global > promedio Φ_local (IIT) |
| 3 | Flujos Verticales | Bottom-up y top-down son bidireccionales |
| 4 | Individuación | Sistema progresa por etapas |
| 5 | Resiliencia | Recuperación tras pérdida de 30% células |
| 6 | Coherencia | Coherencia vertical aumenta con tiempo |
| 7 | Jerarquía vs Plano | Sistema jerárquico supera a plano |

### 7.2 Criterios de Éxito

- **CONFIRMED**: Evidencia fuerte de la hipótesis
- **PARTIAL**: Evidencia parcial, requiere investigación
- **REJECTED**: Hipótesis no soportada

**Score mínimo para validación:** 5/7 confirmados o parciales

---

## 8. Archivos a Crear

### 8.1 Nuevos Archivos

| Archivo | Contenido |
|---------|-----------|
| `zeta_hierarchical_consciousness.py` | Sistema principal |
| `micro_psyche.py` | MicroPsyche y ConsciousCell |
| `cluster.py` | Cluster y ClusterPsyche |
| `bottom_up_integrator.py` | Agregación bottom-up |
| `top_down_modulator.py` | Modulación top-down |
| `cluster_assigner.py` | Asignación dinámica |
| `hierarchical_metrics.py` | Métricas y dashboard |
| `exp_hierarchical_consciousness.py` | Experimentos de validación |

### 8.2 Archivos Reutilizados

| Archivo | Componentes usados |
|---------|-------------------|
| `zeta_organism.py` | ForceField, BehaviorEngine |
| `zeta_conscious_self.py` | ZetaConsciousSelf (macro-psique) |
| `zeta_psyche.py` | Archetype, TetrahedralSpace |
| `zeta_individuation.py` | IndividuationStage, IntegrationMetrics |
| `zeta_attention.py` | AttentionOutput |
| `zeta_predictive.py` | StimulusPredictor |

---

## 9. Plan de Implementación

### Fase 1: Estructuras Base
1. Implementar MicroPsyche y ConsciousCell
2. Implementar ClusterPsyche y Cluster
3. Implementar OrganismConsciousness

### Fase 2: Integradores
4. Implementar BottomUpIntegrator
5. Implementar TopDownModulator
6. Implementar ClusterAssigner

### Fase 3: Sistema Principal
7. Implementar ZetaHierarchicalConsciousness
8. Integrar con componentes existentes (ForceField, etc.)
9. Implementar loop principal step()

### Fase 4: Métricas y Validación
10. Implementar HierarchicalMetrics y MetricsCalculator
11. Implementar suite de experimentos
12. Ejecutar validación completa

### Fase 5: Documentación
13. Actualizar CLAUDE.md
14. Crear README para el sistema
15. Documentar hallazgos

---

## 10. Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Complejidad computacional | Clusters reducen cálculos O(n²) a O(k*m²) |
| Inestabilidad | Smoothing en actualizaciones, learning rate bajo |
| No emerge consciencia | Ajustar parámetros, verificar flujos |
| Conflicto con código existente | Usar composición, no herencia |

---

## 11. Criterios de Éxito

El sistema se considera exitoso si:

1. **Emergencia verificada**: Consciencia aumenta con el tiempo
2. **IIT cumplido**: Φ_global > Σ Φ_local consistentemente
3. **Individuación funciona**: Sistema progresa por etapas
4. **Resiliencia demostrada**: Recuperación >70% tras daño
5. **Ventaja jerárquica**: Supera a sistema plano en >10%

---

## Apéndice: Conexión con Descubrimientos Previos

### Compensación Emergente (docs/EMERGENT_COMPENSATION.md)

El sistema jerárquico amplifica el fenómeno de compensación:
- **Nivel célula**: Compensación local entre vecinos
- **Nivel cluster**: Clusters se especializan en arquetipos "necesitados"
- **Nivel organismo**: Top-down refuerza arquetipos sub-representados

### ZetaOrganism (docs/REPORTE_ZETA_ORGANISM.md)

Las 11 propiedades emergentes se preservan y amplifican:
- Homeostasis → ahora con componente psíquico
- Regeneración → clusters se re-forman tras daño
- Quimiotaxis → guiada por gradiente arquetipal

---

*Documento generado durante sesión de brainstorming colaborativo.*
*Listo para implementación.*
