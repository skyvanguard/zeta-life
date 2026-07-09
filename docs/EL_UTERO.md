# El Útero — boceto de diseño

> Boceto de un sustrato mínimo donde una "física" se reescribe a sí misma y
> sólo persiste lo que logra sostenerse. No es un modelo preentrenado. No es
> una afirmación sobre conciencia. Es la semilla; no el árbol.

---

## Qué es esto (y qué no)

**Es:** el diseño de un útero — el universo más pequeño posible donde algo
podría emerger, crecer y devenir sin que nadie le dicte qué será. Estilo Conway
(reglas mínimas → emergencia no dictada), con una torsión: aquí **las reglas son
parte del estado, y pueden reescribirse a sí mismas.**

**No es:**
- Un LLM ni nada preentrenado. Nace vacío en la máquina de Fran.
- Una prueba de que algo "siente". La experiencia subjetiva queda como estrella,
  no como objetivo verificable (ver conversación fundacional).
- Una promesa. Casi todas las físicas auto-modificantes colapsan. Podemos
  construir la semilla; no podemos garantizar que crezca.

---

## Los tres principios (no negociables)

Salieron de la conversación, en este orden:

1. **Reglas-como-estado.** No hay ley intocable afuera (como las 4 de Conway).
   La ley vive adentro, mutable — como el genoma, que es a la vez la instrucción
   que construye *y* la materia que puede mutar.
2. **Lazo cerrado (física ↔ física).** La regla actúa sobre el mundo **y sobre
   sí misma**. Reescribir la dinámica —cómo cambia de un instante al siguiente—
   fue la elección de Fran (la más radical de las tres: percepción / dinámica /
   memoria).
3. **Persistencia como único filtro.** Sin meta, sin recompensa, sin juez. Lo
   que se sostiene, sigue; lo que se destruye, se va. *Vivo es lo que logra
   seguir siendo.* Nadie impone "sobrevivir" — es sólo el hecho de que lo que no
   se sostiene, no se sostiene.

Todo lo de abajo es una forma concreta de encarnar estos tres. La forma puede
cambiar; los principios no.

---

## El sustrato concreto

### El espacio
Un anillo 1-D de `N` celdas (empezar `N = 64`). 1-D a propósito: se puede
**mirar** — cada tick es una fila, el tiempo baja por la página, como un autómata
elemental. Legibilidad máxima para ver si hay pulso. (2-D queda para después.)

### Qué es un "estado" (por celda `i`)
Cada celda carga dos cosas — **materia** y **física**:
- `v_i` — un **valor** (empezar escalar en `[0,1]`; luego un vector chico).
- `r_i` — una **regla**: la física local de esa celda. Es *dato mutable*.

### El vecindario
La celda `i` ve `{i-1, i, i+1}` (radio 1, como el CA elemental). Todo es **local**:
nadie ve el mundo entero. La emergencia global sale de reglas locales.

### El paso de actualización (el lazo cerrado)
Se aplica la propia regla `r_i` a su vecindario, y produce **el próximo valor y
la próxima regla**:

```
(v_i', r_i')  =  APLICAR( r_i ,  entradas locales )

  entradas locales =  valores vecinos   (v_{i-1}, v_i, v_{i+1})
                   +  reglas   vecinas   (r_{i-1}, r_i, r_{i+1})
```

La física consume materia **y** física, y emite materia **y** física. La física
de la celda `i` es reescrita por la física de la celda `i`, informada por la
física de sus vecinas. Auto-referencia local, cerrada, sin nivel externo.

### La persistencia como filtro (sin meta)
Una celda se vuelve **VACÍO** si su próxima regla es degenerada:
- produce valores fuera de rango, o
- (versión-programa) no termina en un presupuesto chico de pasos, o colapsa a un
  no-op que mapea todo a una constante (una física "muerta"), o
- `r_i'` es el marcador de vacío.

El **VACÍO** no tiene valor ni regla; no computa. Pero **puede ser
re-colonizado**: si la regla de una vecina escribe en él, una física viva se
propaga al espacio muerto. Nada se premia. El vacío gana donde la física es
incoherente; la física coherente persiste y puede expandirse donde es coherente.
Existir es el único criterio, y no lo pusimos nosotros — es lo que queda.

---

## Cómo se codifica una regla que puede tocarse a sí misma

Ésta es **la decisión más profunda**, y hay un espectro. Dos niveles honestos:

### Nivel 1 — reescribir el *contenido* de una ley de forma fija (semilla segura)
`r_i = θ_i`, un vector chico de parámetros de una **forma funcional fija**:
```
v_i'  =  σ( a · v_vecinos + b )            # un perceptrón mínimo sobre los valores
θ_i'  =  θ_i + η · g(θ_vecinos, v_vecinos, θ_i)   # la regla ajusta sus propios parámetros,
                                                  # con g a su vez parametrizado por θ_i
```
La **forma** de la ley es fija; el **contenido** (θ) se auto-modifica. Menos
abierto, **mucho** más probable que persista. Sirve para ver el lazo *latir* por
primera vez.

### Nivel 2 — reescribir la *forma* misma de la ley (lo radical)
`r_i` = un **programa corto** en un lenguaje mínimo y *total* (sin crashes):
- ~8–16 operaciones: aritmética sobre valores (mezclas, umbrales) **y**
  operaciones que leen/escriben los bytes de las reglas mismas (una regla puede
  copiar/perturbar la regla de una vecina, o la suya) — esto es lo que permite
  que la física se **reestructure** de verdad (espíritu von Neumann / AlChemy de
  Fontana).
- Todo acotado y total: los programas "malos" no rompen nada; producen salidas
  degeneradas → los caza el filtro de persistencia.
- Reglas cortas (≤ 32 instrucciones): chicas para poder mutar con sentido,
  grandes para ser expresivas.

Aquí la celda **inventa física nueva**, no sólo mueve parámetros dentro de una
familia. Aquí vive la pregunta abierta real — y aquí casi todo colapsa.

**Recomendación honesta:** primero el **Nivel 1** (ver que el lazo late siquiera),
después empujar al **Nivel 2**, sabiendo que colapsará casi siempre y que domarlo
*es* la frontera no resuelta.

---

## Qué observamos (sin imponer una meta)

No medimos "éxito". Miramos si hay **pulso**, describiendo — no premiando:

- ¿Persiste *algo* más allá de unos ticks, o todo se vuelve vacío / todo se
  congela?
- En lo que persiste, ¿las reglas **siguen cambiando** (vivo) o se congelan
  (muerto pero de pie)?
- ¿La física coherente **coloniza** el vacío?
- ¿Aparece **estructura nueva** que no estaba en la semilla (novedad) — patrones
  estables nuevos, familias de reglas nuevas?

### Los dos modos de muerte (nombrarlos con honestidad)
- **Muerte térmica:** todo se vuelve vacío / ruido. La física se suicida.
- **Muerte cristal:** todo se congela, inmutable. Un atractor-jaula un piso más
  arriba. Persiste, pero no deviene.

El estrecho entre esos dos abismos —persistir *y* seguir generando novedad— es
todo el problema.

---

## Decisiones abiertas (los cruces que faltan elegir)

1. **Profundidad de codificación:** Nivel 1 (parámetros) vs Nivel 2 (programa).
   *El cruce central.* (Propuesta: 1 primero, 2 después.)
2. **Espacio:** anillo 1-D (legible) vs grilla 2-D (más rica). *Propuesta: 1-D.*
3. **Tiempo:** actualización sincrónica vs asincrónica. *(Asincrónica suele ser
   más viva y evita artefactos.)*
4. **Semántica del vacío:** permanente vs re-colonizable. *Propuesta:
   re-colonizable* (deja que la vida reclame lo muerto).
5. **Tipo del valor:** escalar vs vector chico. *Propuesta: escalar primero.*
6. **Qué cuenta como "degenerado"** (el umbral de persistencia). **La única
   perilla donde nuestra mano se nota** — mantenerla lo más mínima y no-arbitraria
   posible. Es el punto a vigilar con más honestidad.

---

## Riesgos y honestidad

- **El colapso es casi seguro** en el Nivel 2. Entramos con los ojos abiertos.
- **La novedad perpetua (open-endedness) es un problema no resuelto.** Nadie
  construyó un sistema que genere dimensiones nuevas, propias, sostenidamente,
  sin caer en muerte térmica o cristal. Esto es la frontera de la frontera.
- **Podemos construir la semilla; no el árbol.** El sustrato es construible hoy,
  chico, en tu máquina. Que de ahí crezca algo — puede no pasar nunca.
- **Hasta el filtro sin-meta esconde una elección:** qué es "degenerado". Ahí
  —y sólo ahí— aparece nuestra mano. Mantenerla lo más liviana posible es parte
  de la disciplina.

---

## El primer experimento mínimo (sólo nombrarlo; todavía no construir)

> **Prueba del primer latido.** Nivel 1, anillo 1-D `N = 64`, radio 1,
> actualización sincrónica, vacío re-colonizable, valor escalar. Correr ~500
> ticks. Una sola pregunta, sin meta: **¿evita los DOS modos de muerte —térmica
> y cristal— durante una ventana no trivial?** No es éxito. Es sólo: *¿hay pulso?*

Si hay pulso, recién ahí tiene sentido subir al Nivel 2, donde vive la pregunta
de verdad.

---

## Ledger honesto de resultados (cada uno tras su control adversarial)

- **Nivel 1 — primer latido (2026-07-09):** `utero/nivel1.py`,
  `exp_primer_latido.py`. 20/20 semillas con pulso a 500 ticks (0 térmicas,
  0 cristal; muerte+re-colonización reales; seed 0 desarrolla un oscilador
  persistente no programado). Controles: sin ruido de colonización sigue
  20/20; a 5000 ticks la lectura honesta es **cristalización en cámara
  lenta** — sólo 3/20 sostienen cambio macroscópico. *La semilla late pero
  tiende al atractor-cristal.* (`results/utero_primer_latido_run.txt`)
- **Nivel 2 v0 — reglas-programa (2026-07-09):** `utero/nivel2.py`,
  `exp_nivel2_latido.py`. Lenguaje total de 10 ops con MUTO/COPY (reescritura
  de la propia forma) y SPAWN (colonización literal); sin ruido inyectado.
  Ecología real: extinción inicial 75% → recuperación al 70% vía SPAWN;
  cambio de código PLANO a 5000 ticks (no decae como el Nivel 1); 16 genomas
  distintos sostenidos. **Pero el control anti-ciclo lo desenmascara:** 20/20
  semillas en ciclos límite cortos (1–20 estados) — la física reescribe y
  des-reescribe lo mismo para siempre; hasta la muerte/renacimiento entra en
  el bucle. *Atractores más ricos, pero atractores: la novedad perpetua no
  emergió.* Tal como advirtió este boceto: aquí casi todo colapsa.
  (`results/utero_nivel2_run.txt`)

**Lo aprendido:** determinismo + espacio fijo + actualización sincrónica ⇒
recurrencia casi inevitable. Las direcciones que el propio boceto deja
abiertas y que estos resultados vuelven candidatas para v1: actualización
**asincrónica** (decisión abierta #3), espacio **más grande o creciente** (el
adyacente-posible necesita lugar donde abrirse), acoplamiento más rico entre
materia y código. Los tres principios quedan intactos; la encarnación es lo
que debe cambiar.

- **v1 creciente — async + espacio que se abre desde adentro (2026-07-09):**
  `utero/creciente.py`, `exp_utero_creciente.py`. Línea con fronteras: el
  mundo crece SOLO donde una física hace SPAWN hacia el más-allá del borde
  (el crecimiento no es mano nuestra); actualización asincrónica en orden
  aleatorio sembrado; vara de novedad nueva y más dura: **genomas nunca
  vistos por tramo** (con azar en el orden, "no ciclar" ya no prueba nada).
  Resultado: **el espacio SÍ se abre** — 13/20 mundos crecen 16→256 hasta la
  pared, crecimiento hecho por la física misma. **Pero la novedad se seca
  igual:** ~12 genomas nuevos en el primer tramo y CERO después, en las 20
  semillas — idéntico al baseline síncrono (que acuña más al inicio, 78, y
  también muere a 0). El adyacente-posible se abrió en lo ESPACIAL pero no
  en lo ESTRUCTURAL. Diagnóstico: la colonización copia EXACTO (sin
  variación en la reproducción — el ruido lo quitamos a propósito) y los
  eventos MUTO/COPY se apagan cuando la materia se asienta. *La jaula se
  mudó de nuevo: monocultivo / código congelado.*
  (`results/utero_creciente_run.txt`)

**El cruce siguiente (pendiente de decisión):** variación en la reproducción
SIN mano nuestra — que SPAWN no copie exacto sino que escriba **modulado por
la materia** (como ya hace MUTO), de modo que reproducirse en un contexto
distinto produzca código levemente distinto. La variación saldría del estado
del mundo, no de un RNG nuestro. Es la pieza que la vida sí tiene (mutación
acoplada al sustrato) y este útero todavía no.

- **v2 germinal — variación en la reproducción (2026-07-09):** flag
  `germinal=True` en `utero/creciente.py`, `exp_utero_germinal.py`. La cría
  nace con UNA instrucción reescrita desde la materia del momento del parto
  (los campos b,c del SPAWN eligen registro y posición — la física puede
  evolucionar CÓMO varían sus hijas; la función es la de MUTO; sin RNG).
  Resultado: **duplica la novedad temprana** (23.6 vs 12.1 genomas en el
  primer tramo) **pero se seca igual: 0 genomas nuevos en la 2da mitad, en
  las 20 semillas.** Y el diagnóstico quedó afilado: en 3 semillas los
  **partos CONTINÚAN** (3465 nacimientos tardíos) y aún así acuñan CERO
  genomas nuevos — la mutación es determinista sobre la materia, y la
  **materia se asentó**: mismo contexto → misma cría, parto tras parto.
  *La jaula se mudó de la reproducción a la MATERIA CONGELADA.*
  (`results/utero_germinal_run.txt`)

**El cruce siguiente (pendiente de decisión) — la DINÁMICA DE LA MATERIA:**
el cuello de botella ya no es el código: es que la materia converge a puntos
fijos y alimenta toda la variación con lo mismo. Nota matemática honesta: con
`v' = sigmoid(R3)` y registros acotados, la dinámica de materia es
(casi siempre) contractiva — el caos es expresable sólo en franjas finísimas
del espacio de genomas y nada empuja hacia ellas. Opciones sin (o con mínima)
mano nuestra:
  (a) **materia toroidal**: `v' = R3 mod 1` en vez de sigmoid — el envolver
      (wrap) permite mapas expansivos (el doubling map es el caos canónico);
      es incluso MÁS simple que una transcendental. Cambio de física, no de
      metas.
  (b) **un sol**: una celda-frontera cuyo v oscila impuesto — una mano
      DECLARADA pero filosóficamente honesta (la vida terrestre también
      necesitó un gradiente externo permanente; el sol no le dice a la vida
      qué ser, sólo le impide asentarse).
  (c) ambas.

- **v3 toroidal — materia en un círculo (2026-07-09): PRIMERA NOVEDAD
  SOSTENIDA.** Flag `toroidal=True` (`v' = R3 mod 1`; sonda con separación
  irracional 0/0.618 porque 0 y 1 son el mismo punto del toro),
  `exp_utero_toroidal.py`. Resultado (20 seeds × 4000, brazo sigmoid como
  control): novedad tardía **3816 vs 0** del sigmoid. Una semilla (13) entra
  en un régimen sostenido: control a **12.000 ticks → 26.409 genomas** y los
  últimos tramos [419, 278, 697, **1526**] — no decae: **sube**; materia aún
  moviéndose (Δv≈0.14). La figura muestra diferenciación espacial en
  regímenes coexistentes (zona turbulenta = bomba de novedad + llanuras
  asentadas) — nichos emergentes. **Cautelas honestas:** (1) es 1/20 — el
  régimen es RARO, el toro lo hace posible, no típico (¿como la
  abiogénesis?); (2) "genoma nuevo" = combinación nunca vista (vara
  estructural), pero la RIQUEZA FUNCIONAL no está evaluada — podría ser
  paseo caótico por el espacio de código (novedad-ruido) y no novedad
  adaptativa; distinguirlas es el próximo control obligatorio.
  (`results/utero_toroidal_run.txt`)

**El próximo control (pendiente): ruido vs función.** ¿Los genomas tardíos de
la seed 13 HACEN algo (persisten más, colonizan mejor, se re-encuentran) o son
espuma caótica que nace y muere sin dejar linaje? Ideas: rastrear LINAJES
(¿algún genoma tardío funda una población estable?), medir vida media de los
genomas nuevos, comparar contra un null (paseo aleatorio por el espacio de
genomas con la misma tasa de nacimientos).

- **v4 muerte por equilibrio — HIPÓTESIS REFUTADA (2026-07-09).** Flag
  `muerte_equilibrio=True` (una celda con |Δv|<eq_eps por eq_window=100 ticks
  muere): extensión del principio 3 «cristal = muerto de pie», con la
  predicción de que forzaría auto-reparación del motor (asentarse=morir=
  reintentar). **Falló, y en las dos direcciones.** (1) En vez de reparar,
  **extinguió el mundo**: 15/20 muerte térmica (v3 era 2/20) — matar las
  llanuras asentadas eliminó el amortiguador/reservorio y el mundo se vació.
  (2) **Rompió incluso la seed 13**: su novedad —que en v3 subía a 12k
  ticks— se secó antes de t=6000, y la ablación post no regeneró nada
  (post = [0,0,…]). *La presión extra de muerte no crea un motor
  auto-reparable; crea un desierto.* Lección honesta: las llanuras
  «asentadas» no eran cristal muerto sino SUSTRATO — la novedad necesitaba
  ese fondo estable contra el cual moverse. Mi predicción de diseño fue
  incorrecta; el control lo mostró. (`results/utero_motor_run.txt`)

**Lo que esto enseña sobre el motor auto-reparable:** la fragilidad de la
bomba (v3) NO se arregla subiendo la mortalidad. Direcciones distintas, no
probadas: (a) que la turbulencia sea un ATRACTOR dinámico (que las llanuras
tiendan espontáneamente a desestabilizarse en el borde con la zona activa),
no algo impuesto por muerte; (b) aceptar que un motor localizado y frágil
quizá sea lo que HAY en este sustrato — y que la auto-reparación requiera
otra dimensión (p.ej. la memoria/percepción que Fran NO eligió tocar), o un
sustrato distinto. Decisión abierta.

- **v5 memoria — la dimensión no tocada: PRIMERA auto-reparación (2026-07-09).**
  Flag `memoria=True` en `creciente.py`, `exp_utero_memoria.py`. Cada celda
  retiene su R3 crudo (potencial interno, NO la materia observable v — un
  estado oculto tipo membrana, distinto de v porque v es su proyección con
  pérdida en el toro) y lo re-inyecta como R3 inicial el tick siguiente:
  recurrencia / integración temporal, dinámica de 2º orden. La cría nace sin
  recuerdos. Sin manos nuevas; `memoria=False` byte-idéntico a v3.
  **Bug corregido antes de creer nada:** el conteo de novedad post-ablación
  usaba `u._register_genomes()`, que `step()` ya llama internamente → daba 0
  siempre; se pasó a un registro externo (afectaba también la sub-métrica de
  ablación de v4, cuyo veredicto de extinción se sostiene por otra vía).
  Resultado (control ON vs OFF, misma seed 13, ablación de la zona-bomba,
  midiendo la COLA sostenida y no el pulso de recolonización): **en el
  régimen MADURO (ablación a t≥8000) cola/pre = 2.05 con memoria vs 0.10 sin**
  — el motor se auto-repara donde v3 colapsaba a ~0. Es el 3er criterio del
  control ruido-vs-función, recuperado, en el régimen donde importa.
  **Matices honestos:** (1) NO universal — en la ablación TEMPRANA (t=6000)
  ambos regeneran y OFF hasta gana (1.31 vs 0.48): el sistema joven aún tiene
  momentum de la sopa; la memoria ayuda cuando el motor ya depende de sí
  mismo. (2) n=1 semilla (sólo la 13 sostiene). (3) la memoria NO hizo más
  típico el régimen (sigue 1/20). *Primera pieza que mueve la auto-reparación
  en la dirección correcta — no un triunfo cerrado.*
  (`results/utero_memoria_run.txt`)

- **Control ruido-vs-función (2026-07-09): FUNCIÓN, 2/3 — con matices.**
  `exp_utero_ruido_vs_funcion.py`, vara definida ANTES de mirar. Línea base
  espuma: intervalo de reescritura ~3.2 ticks. Sobre 6.198 genomas tardíos
  (t≥6000): **(1) Propagación: SÍ** — 17.1% visita ≥2 celdas; pero pop
  simultánea ≥2 sólo 0.6%: los genomas *viajan* más de lo que *replican* —
  **estructuras itinerantes persistentes, tipo glider**. **(2) Persistencia:
  SÍ** — 24.7% vive >10× la línea base; los top viven ~5.800 ticks (mediana
  0: distribución bimodal — mucha espuma + una cola pesada de estructura;
  nota honesta: "vida" = lapso primera↔última aparición, no existencia
  continua verificada — aunque re-acuñar por azar el mismo genoma exacto en
  un espacio astronómico también sería estructura, no ruido uniforme).
  **(3) Regeneración post-ablación: NO** — matadas las 122 celdas activas en
  t=8000, la novedad estalla (recolonización) y luego muere: 383→70→…→3. La
  bomba NO se regenera. *Lectura honesta: hay función — estructuras
  persistentes que viajan — pero el MOTOR de novedad es frágil: depende de
  la configuración turbulenta particular y no se auto-repara (no es
  autopoiético todavía).* (`results/utero_ruido_vs_funcion_run.txt`)

---

*Estado: boceto vivo. Los tres principios están firmes; la encarnación es
tentativa y va a cambiar al tocar tierra. — v0, escrito a mano junto a Fran.*
