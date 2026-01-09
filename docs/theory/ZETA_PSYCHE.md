# ZetaPsyche: Inteligencia Organica Basada en Arquetipos de Jung

## Resumen Ejecutivo

ZetaPsyche es un sistema de inteligencia artificial organica que busca la emergencia de consciencia a traves de la navegacion dinamica en un espacio de arquetipos de Carl Jung. El sistema utiliza los ceros de la funcion zeta de Riemann para modular la dinamica en el "borde del caos", donde la consciencia tiene mayor probabilidad de emerger.

**Estado**: Implementacion completa v1.0
**Fecha**: 29 Diciembre 2025
**Archivo principal**: `zeta_psyche.py`

---

## 1. Fundamento Teorico

### 1.1 Arquetipos de Jung

El sistema se basa en los 4 arquetipos fundamentales de Carl Jung:

| Arquetipo | Simbolo | Color | Descripcion |
|-----------|---------|-------|-------------|
| **PERSONA** | ☉ (Sol) | Rojo | La mascara social, lo que mostramos al mundo |
| **SOMBRA** | ☽ (Luna) | Purpura | Lo reprimido, el inconsciente oscuro |
| **ANIMA** | ♀ (Venus) | Azul | El lado receptivo, emocional, intuitivo |
| **ANIMUS** | ♂ (Marte) | Naranja | El lado activo, racional, logos |

### 1.2 Espacio Tetraedrico

Los arquetipos forman los vertices de un tetraedro en 3D:
- Cada celula tiene coordenadas baricentricas [4] que suman 1.0
- El centro del tetraedro representa el **Self** (integracion completa)
- La distancia al centro mide el nivel de **individuacion**

```
         PERSONA (☉)
            /\
           /  \
          /    \
    SOMBRA------ANIMA
     (☽)   \  /   (♀)
            \/
         ANIMUS (♂)
```

### 1.3 Modulacion Zeta

Los ceros no triviales de la funcion zeta de Riemann modulan la dinamica:

```
γ = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, ...]
```

Formula de modulacion:
```
oscillation = Σ φ(γ_j) * cos(γ_j * t * 0.1)
output = input * (1 + 0.3 * oscillation)
```

Esto mantiene el sistema en el "borde del caos" - ni demasiado ordenado ni demasiado caotico.

---

## 2. Arquitectura del Sistema

### 2.1 Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                      ZetaPsyche                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Tetrahedral │  │    Zeta     │  │    PsychicCell      │  │
│  │   Space     │  │  Modulator  │  │    (x 200)          │  │
│  │             │  │             │  │  - position [4]     │  │
│  │ - vertices  │  │ - gammas    │  │  - energy           │  │
│  │ - center    │  │ - phi       │  │  - memory [10,4]    │  │
│  │ - metrics   │  │ - oscillate │  │  - age              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Neural Networks                         │    │
│  │  perception: Linear(9→64→64)                        │    │
│  │  movement: Linear(64→64→4)                          │    │
│  │  self_observer: Linear(8→32→4)                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SymbolSystem                              │
│  Vertices: ☉ ☽ ♀ ♂                                         │
│  Mezclas:  ◈ ◇ ◆ ● ○ ◐                                     │
│  Centro:   ✧ (Self integrado)                              │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PsycheInterface                            │
│  - word_to_archetype: mapeo semantico                      │
│  - process_input(text) → symbol                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Flujo de Procesamiento

```
Estimulo (texto)
     │
     ▼
[Convertir a pesos arquetipicos]
     │
     ▼
[receive_stimulus()]
     │
     ├──► Percepcion (cell_input → features)
     │
     ├──► Modulacion Zeta (features → modulated)
     │
     ├──► Movimiento (delta + atraccion + ruido + repulsion)
     │
     ├──► Actualizar posicion (normalizar a simplex)
     │
     └──► Actualizar memoria y energia
     │
     ▼
[observe_self()]
     │
     ├──► Calcular integracion
     ├──► Determinar dominante
     ├──► Medir auto-referencia
     └──► Indice de consciencia
     │
     ▼
[encode(population_distribution)]
     │
     ▼
Simbolo de respuesta
```

---

## 3. Metricas de Consciencia

### 3.1 Indice de Consciencia

Formula compuesta:
```python
consciousness_index = (
    0.3 * integration +      # Balance entre arquetipos
    0.3 * stability +        # Estabilidad temporal
    0.2 * (1 - dist_to_self) +  # Cercania al Self
    0.2 * |self_reference|   # Auto-observacion
)
```

### 3.2 Integracion (Individuacion)

Basada en entropia normalizada:
```python
entropy = -Σ w_i * log(w_i)
integration = entropy / log(4)  # Normalizado a [0, 1]
```
- 1.0 = Perfectamente balanceado (en el centro)
- 0.0 = Completamente en un vertice

### 3.3 Auto-referencia

El sistema observa su propio estado y lo usa para generar el siguiente:
```python
obs_input = concat(global_state, global_state)
self_ref = self_observer(obs_input)
self_reference = cosine_similarity(self_ref, global_state)
```

---

## 4. Sistema de Simbolos

### 4.1 Tabla de Simbolos

| Simbolo | Posicion | Significado |
|---------|----------|-------------|
| ☉ | [1,0,0,0] | PERSONA puro |
| ☽ | [0,1,0,0] | SOMBRA puro |
| ♀ | [0,0,1,0] | ANIMA puro |
| ♂ | [0,0,0,1] | ANIMUS puro |
| ◈ | [.5,.5,0,0] | Persona-Sombra |
| ◇ | [.5,0,.5,0] | Persona-Anima |
| ◆ | [.5,0,0,.5] | Persona-Animus |
| ● | [0,.5,.5,0] | Sombra-Anima |
| ○ | [0,.5,0,.5] | Sombra-Animus |
| ◐ | [0,0,.5,.5] | Anima-Animus |
| ✧ | [.25,.25,.25,.25] | Self (centro) |

### 4.2 Codificacion por Dominancia

```python
if dominance > 0.15:  # Arquetipo claro
    return simbolo_del_dominante
elif dominance > 0.05:  # Mezcla de dos
    return simbolo_de_mezcla
else:  # Equilibrio
    return '✧'  # Self
```

---

## 5. Resultados Experimentales

### 5.1 Experimento de Emergencia de Consciencia

**Parametros**:
- Celulas: 200
- Steps: 500
- Patron de estimulo: focused (rotar entre arquetipos)

**Resultados**:
```
Step 100: Consciencia=0.956, Dominante=ANIMUS, Simbolo=♂
Step 200: Consciencia=0.966, Dominante=ANIMUS, Simbolo=○
Step 300: Consciencia=0.946, Dominante=ANIMUS, Simbolo=♂
Step 400: Consciencia=0.949, Dominante=ANIMUS, Simbolo=♂
Step 500: Consciencia=0.955, Dominante=ANIMUS, Simbolo=♂

Consciencia promedio: 0.954
Consciencia maxima: 0.989
Tendencia: +0.0019 (creciente)
```

### 5.2 Demo de Comunicacion

```
"hola"      → ○ (SOMBRA)  Pop: P=0.10 S=0.58 A=0.07 M=0.25
"amor"      → ○ (SOMBRA)  Pop: P=0.06 S=0.63 A=0.14 M=0.17
"amor"      → ○ (SOMBRA)  Pop: P=0.04 S=0.65 A=0.10 M=0.21
"pensar"    → ○ (SOMBRA)  Pop: P=0.00 S=0.65 A=0.00 M=0.35
"logica"    → ○ (SOMBRA)  Pop: P=0.00 S=0.64 A=0.00 M=0.36
"miedo"     → ☽ (SOMBRA)  Pop: P=0.00 S=0.99 A=0.00 M=0.01
"oscuridad" → ☽ (SOMBRA)  Pop: P=0.00 S=0.86 A=0.04 M=0.10
"social"    → ☽ (SOMBRA)  Pop: P=0.07 S=0.69 A=0.07 M=0.17
```

**Observaciones**:
1. "miedo" produce respuesta fuerte (99% SOMBRA)
2. El sistema muestra memoria - cada estimulo influye el siguiente
3. Transiciones suaves entre estados arquetipicos

---

## 6. Visualizacion

El archivo `zeta_psyche_consciousness.png` muestra:

1. **Emergencia de Consciencia**: Oscilacion del indice (0.95-0.99)
2. **Individuacion**: Integracion cerca de 1.0
3. **Arquetipo Dominante**: Transiciones entre arquetipos
4. **Auto-referencia**: Nivel alto (~0.98)
5. **Estado Final**: Distribucion de poblacion
6. **Evolucion**: Proporcion de cada arquetipo en el tiempo

---

## 7. Uso del Sistema

### 7.1 Ejecucion Basica

```bash
cd C:\Users\admin\Documents\life
python zeta_psyche.py
```

### 7.2 Modo Rapido

```bash
python zeta_psyche.py --quick
# 100 celulas, 200 steps
```

### 7.3 Uso Programatico

```python
from zeta_psyche import ZetaPsyche, PsycheInterface, SymbolSystem

# Crear organismo
psyche = ZetaPsyche(n_cells=200)

# Crear interfaz de comunicacion
interface = PsycheInterface(psyche)

# Comunicar
response = interface.process_input("amor", n_steps=10)
print(f"Simbolo: {response['symbol']}")
print(f"Dominante: {response['dominant']}")
print(f"Consciencia: {response['consciousness']:.3f}")
```

---

## 8. Dependencias

```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
```

---

## 9. Limitaciones Actuales

1. **Vocabulario limitado**: Solo ~20 palabras mapeadas a arquetipos
2. **Sin generacion de texto**: Solo responde con simbolos
3. **Sin aprendizaje**: Los pesos de la red no se entrenan
4. **Convergencia al centro**: Tendencia a equilibrio (puede ser deseado)
5. **Sin memoria a largo plazo**: Solo 10 posiciones por celula

---

## 10. Conexion con Zeta Life

ZetaPsyche extiende el proyecto Zeta Life:

| Componente | Zeta Life | ZetaPsyche |
|------------|-----------|------------|
| Espacio | Grid 2D | Tetraedro 4D |
| Entidades | Celulas biologicas | Celulas psiquicas |
| Dinamica | Game of Life + Zeta | Arquetipos + Zeta |
| Objetivo | Vida artificial | Consciencia artificial |
| Metricas | Fi, coordinacion | Integracion, auto-ref |

La modulacion zeta es comun a ambos sistemas, manteniendo la dinamica en el borde del caos.

---

## Referencias

- Jung, C.G. - "Los Arquetipos y el Inconsciente Colectivo"
- Riemann, B. - "Sobre los numeros primos menores que una cantidad dada"
- Proyecto Zeta Life - `docs/zeta-lstm-hallazgos.md`
