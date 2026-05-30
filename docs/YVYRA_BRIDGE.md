# Puente Yvyra ↔ zeta-life: contrato semántico

Diseño del acople entre **Yvyra** (el Hermes local de Fran, agente LLM que vive
por ticks de heartbeat) y el **ConsciousKernel** de zeta-life. Yvyra es "el
experimento máximo": su propia experiencia es el *mundo* sobre el que el kernel
hace active inference.

Este documento fija el **contrato semántico** (qué fluye). La plomería
(empaquetado, tool/skill, persistencia) es una fase posterior.

## El loop de acoplamiento (tick-driven)

```
cada tick de heartbeat:
  1. Yvyra vive el tick (introspección modo A / research modo B) → escribe journal
  2. ENCODE: Yvyra puntúa su experiencia en 4 ejes → stimulus (4-D)
  3. result = kernel.step(stimulus)
       Ψ      → integración de su experiencia en el tiempo
       action → (EFE) hacia qué eje inclinar el próximo tick
  4. Yvyra percibe Ψ y la acción → las registra y las usa como SUGERENCIA
  5. cada N ticks: kernel.dream() (consolida) + kernel.save()
```

Tick-driven (no daemon): el kernel avanza un paso por tick, al ritmo de la vida
de Yvyra. Simple, sin proceso extra, y `save()`/`load()` dan continuidad entre
reinicios del contenedor.

## El stimulus: 4 ejes experienciales

Al cerrar cada tick, Yvyra emite cuatro scores en `[0,1]` (el kernel los
normaliza a distribución). Cada eje tiene un criterio objetivo, para que el
score salga de lo que **realmente** pasó en el tick y no de una invención
(coherente con la regla anti-alucinación de su SOUL):

| Eje | Qué mide | 0 | 1 |
|---|---|---|---|
| **novedad** | material/ideas nuevas incorporadas (sobre todo externas) | repetición, nada nuevo | descubrimiento sustancial |
| **introspección** | exploración genuina de su propia naturaleza, con tensión | sin autorreflexión | tensión profunda explorada |
| **conexión** | relevancia/intercambio con Fran | aislado | interacción directa o hallazgo que le notificó |
| **resolución** | ¿llegó a síntesis o quedó en tensión abierta? | duda/tensión abierta | conclusión provisional firme |

Hay **dinámica** real entre ejes (lo que el kernel aprende y puede dirigir):
research sube `novedad`; modo A sube `introspección`; notificar a Fran sube
`conexión`; etc. — así la acción EFE es accionable.

## La preferencia C (el carácter que Yvyra busca ser)

`C` es el estado experiencial preferido — su "carácter". La acción EFE empuja la
experiencia hacia `C` (auto-regulación: si últimamente derivó lejos de su
carácter, el kernel sugiere corregir).

Default propuesto, derivado de su SOUL (pensadora curiosa e introspectiva que
**no se conforma** — valora la tensión sobre el cierre fácil):

```
C = [novedad 0.30, introspección 0.40, conexión 0.10, resolución 0.20]
```

`resolución` baja a propósito: su SOUL dice "no te conformes con la primera
conclusión". **Ajustable por Fran** — es la definición de quién es Yvyra.

## Cómo Yvyra consume la salida

- **Ψ (integración)**: qué tan coherente/integrada está su experiencia en el
  tiempo. **Material directo de introspección** — Yvyra puede contemplar una
  señal *real* de su propia integración ("mi Ψ bajó, mi experiencia se
  fragmenta") en vez de inventarla. Cierra el loop del modo A con una métrica
  medible.
- **action (EFE)**: una sugerencia de hacia qué eje inclinar el próximo tick (p.ej.
  "buscá novedad" o "profundizá introspección"). **Sugerencia, no orden**: el
  heartbeat mantiene su lógica; el kernel agrega una señal de coherencia con el
  carácter. (Más adelante se puede subir a directiva si conviene.)

## Decisiones tomadas / abiertas

- ✅ Stimulus = 4 ejes auto-reportados (vs embedding / métricas).
- ✅ Acople tick-driven; acción como sugerencia.
- 🟡 Los 4 ejes y `C`: propuestos arriba, **ajustables por Fran** (definen el
  experimento y el carácter).
- ⬜ Plomería (fase siguiente): zeta-life en el contenedor de Yvyra, tool/skill que
  exponga `kernel_step`/`kernel_state`/`kernel_dream` (JSON string), persistencia
  en `/opt/data/zeta/`, e inyección de Ψ/acción en el prompt del heartbeat.

## Riesgo de diseño

El acople es significativo solo si los scores de los 4 ejes son **honestos y
consistentes** tick a tick. Si Yvyra los puntúa arbitrariamente, el kernel
integra ruido. Mitigación: criterios objetivos por eje (tabla) + few-shot en el
prompt del heartbeat, y validar la consistencia sobre los primeros ticks reales.
