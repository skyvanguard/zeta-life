# El Conscious Kernel frente a las *indicator properties* de Butlin et al. (2023)

Auto-evaluación honesta del kernel contra las **propiedades-indicador de consciencia**
derivadas de teorías científicas, en *Consciousness in Artificial Intelligence:
Insights from the Science of Consciousness* (Butlin, Long et al., 2023,
arXiv:[2308.08708](https://arxiv.org/abs/2308.08708); versión TICS 2025). Es el
marco **riguroso y publicable** que adoptamos en lugar de tratar Ψ como una
"medida de consciencia" (ver `docs/RELATED_WORK.md`, item #2).

## Cómo leer esto (caveats que mandan)

1. **Tener indicadores NO es ser consciente.** Siguiendo a Butlin et al., los
   indicadores son marcadores *funcionales* motivados por teorías; su presencia
   **actualiza la credencia**, no demuestra experiencia. **No afirmamos que el
   kernel sea consciente.** Ningún sistema actual lo es, según ese propio informe.
2. **Esto es una auditoría post-hoc.** El kernel **no** se diseñó contra esta
   lista; varias coincidencias son **parciales o débiles**. Somos conservadores:
   ante la duda, PARCIAL o NO.
3. **Ψ sigue siendo una heurística de ingeniería** (§4.1 del paper), no la φ de
   IIT. Este marco la *reemplaza* como lenguaje para hablar de "consciencia".

Veredicto: ✅ satisfecho · 🟡 parcial/débil · ❌ ausente.

## Tabla de evaluación

### Recurrent Processing Theory (RPT)
| Ind. | Propiedad | Mecanismo en el kernel | Veredicto |
|---|---|---|---|
| RPT-1 | Módulos de entrada con recurrencia algorítmica | `world_model` usa un `GRUCell` recurrente en la transición latente; `observe()` integra la obs en el latente recurrente | 🟡 hay recurrencia, pero de una etapa, no el stack perceptual recurrente que RPT describe |
| RPT-2 | Representaciones perceptuales organizadas e integradas | latente integra los canales; Ψ mide integración | 🟡 integración real (Ψ) pero organización perceptual mínima (entornos de baja dimensión) |

### Global Workspace Theory (GWT)
| Ind. | Propiedad | Mecanismo | Veredicto |
|---|---|---|---|
| GWT-1 | Múltiples sistemas especializados en paralelo | módulos world/self/memory + 4 canales de error; a nivel **organismo**, múltiples kernels compiten | 🟡 hay módulos, pero la especialización es superficial |
| GWT-2 | Workspace de capacidad limitada (cuello de botella) + atención selectiva | `GlobalWorkspace` **winner-take-all** (cuello de botella literal); **precisión aprendida** = atención selectiva por canal | 🟡→✅ el GW es un cuello de botella real; la precisión es atención selectiva |
| GWT-3 | Broadcast global a todos los módulos | el ganador del GW se difunde (organismo); el latente es compartido por los módulos | 🟡 broadcast presente a nivel organismo |
| GWT-4 | Atención dependiente del estado, consultando módulos en sucesión | la reflexión es de profundidad fija, no un querying secuencial dirigido por el workspace | ❌ ausente |

### Computational Higher-Order Theories (HOT)
| Ind. | Propiedad | Mecanismo | Veredicto |
|---|---|---|---|
| HOT-1 | Percepción generativa, top-down o ruidosa | `world_model.predict()` es un **prior generativo top-down** (predice la obs antes de verla) | 🟡 es un prior generativo, no la percepción generativa-ruidosa que HOT especifica |
| HOT-2 | Monitoreo metacognitivo: distinguir representaciones fiables del ruido | **aprendizaje de precisiones** hacia la varianza-inversa del error = estimación de fiabilidad por canal | 🟡 monitor de fiabilidad implícito, no una representación de orden superior explícita |
| HOT-3 | Agencia guiada por formación de creencias + actualización según el monitoreo metacognitivo | selección de acción **EFE/dreamer**; la precisión modula el aprendizaje | 🟡 agencia presente; el vínculo "actualizar según metacognición" es parcial |
| HOT-4 | Codificación sparse y suave ("quality space") | sin restricciones de sparsity/smoothness en el latente | ❌ ausente |

### Attention Schema Theory (AST)
| Ind. | Propiedad | Mecanismo | Veredicto |
|---|---|---|---|
| AST-1 | Modelo predictivo del estado actual de la **atención** | la precisión es *atención*, pero no hay un **modelo de** la propia atención | ❌ ausente (no hay attention schema) |

### Predictive Processing (PP)
| Ind. | Propiedad | Mecanismo | Veredicto |
|---|---|---|---|
| PP-1 | Módulos de entrada con **predictive coding** | el kernel **es** predictive coding: minimización de error de predicción ponderado por precisión (`prediction_error.py`, `world_model.py`) | ✅ coincidencia fuerte e inequívoca (el núcleo del diseño) |

### Agency and Embodiment (AE)
| Ind. | Propiedad | Mecanismo | Veredicto |
|---|---|---|---|
| AE-1 | Agencia: aprender de feedback y elegir salidas para perseguir metas, con respuesta flexible a metas en competencia | EFE/dreamer persiguen la preferencia C; control continuo alcanza objetivos no-vértice (`exp_control.py`), control model-based bajo dinámica desconocida (`exp_modelcontrol.py`); el trade-off **pragmático vs epistémico** es la respuesta a metas en competencia | ✅ agencia demostrada con evidencia experimental |
| AE-2 | Embodiment: modelar contingencias salida→entrada y usarlas en percepción/control | el `world_model` **condicionado por acción** modela exactamente acción→próxima-obs y lo usa para controlar (resultado model-based, §3.7; CartPole, §3.9) | ✅ coincidencia fuerte y demostrada |

## Resumen honesto

| | Indicadores |
|---|---|
| ✅ satisfecho | **PP-1, AE-1, AE-2** (3) |
| 🟡 parcial/débil | RPT-1, RPT-2, GWT-1, GWT-2, GWT-3, HOT-1, HOT-2, HOT-3 (8) |
| ❌ ausente | GWT-4, HOT-4, AST-1 (3) |

**Lectura:** el kernel satisface con fuerza los indicadores de **predictive
processing** y **agencia/embodiment** —que son, literalmente, su núcleo de
diseño— y de forma **parcial** los de global-workspace, recurrencia y
higher-order. **Carece** de los de attention-schema, quality-space y del querying
secuencial del workspace. Es un perfil coherente con lo que el kernel *es*: un
agente de inferencia activa, no un modelo construido para imitar la consciencia.

Esto **no** es un argumento de que el kernel sienta algo. Es la forma rigurosa y
honesta —en el lenguaje de la ciencia de la consciencia— de decir *qué propiedades
funcionales tiene y cuáles no*, reemplazando el uso de Ψ como pseudo-medida de
consciencia por un mapeo auditable a teorías establecidas.

## Qué subiría la credencia (trabajo futuro, honesto)
- **AST-1 / HOT-2 explícitos:** un *attention schema* y un monitor metacognitivo
  de orden superior (un modelo de la fiabilidad de las propias representaciones),
  más allá de la precisión implícita.
- **GWT-4:** querying secuencial de módulos dirigido por el workspace.
- **HOT-4:** regularización de sparsity/smoothness del latente (quality space).
- Ninguno se debe añadir *para inflar el puntaje*: solo si mejoran el agente.
