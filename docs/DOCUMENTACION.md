# Zeta Game of Life - Documentacion Completa

## Proyecto: Integracion de Kernels Riemann-Zeta en Automatas Celulares

**Autor del Framework Teorico**: Francisco Ruiz
**Fecha de Implementacion**: 26 Diciembre 2025

---

## 1. Fundamento Teorico

### 1.1 Hipotesis de Riemann y Ceros de Zeta

La funcion zeta de Riemann:
```
zeta(s) = sum_{n=1}^{inf} 1/n^s
```

Tiene ceros no triviales en la linea critica `s = 1/2 + i*gamma`, donde gamma son las partes imaginarias:
- gamma_1 = 14.134725...
- gamma_2 = 21.022040...
- gamma_3 = 25.010858...
- ...

### 1.2 Kernel Zeta (del Paper)

El kernel central del framework:
```
K_sigma(t) = sum_rho exp(-sigma * |gamma|) * (exp(i*gamma*t) + exp(-i*gamma*t))
           = 2 * sum_rho exp(-sigma * |gamma|) * cos(gamma * t)
```

**Propiedades**:
- Decay exponencial controlado por sigma (regularizacion Abel)
- Oscilaciones caracteristicas de los ceros
- Correlaciones de largo alcance: O(1/log|t|)

### 1.3 Transformada de Laplace Bilateral

```
L_zeros(s) = sum_rho (1/(s - rho) + 1/(s - conjugado(rho)))
```

Implementa memoria temporal con propiedades espectrales unicas.

---

## 2. Implementacion por Fases

### Fase 1: Inicializacion con Ruido Zeta Estructurado

**Archivo**: `zeta_game_of_life.py`

**Concepto**: Reemplazar la inicializacion aleatoria clasica con un campo estructurado basado en superposicion de ondas de los ceros de zeta.

**Algoritmo**:
```python
field[i,j] = sum_k w_k * exp(i * (gamma_k * x + gamma_k' * y + phi_k))
```
donde:
- w_k = exp(-sigma * |gamma_k|)
- phi_k = fase aleatoria para romper simetria
- gamma_k' = gamma con offset para direccion y

**Resultados Fase 1**:
| Metrica | Aleatorio | Zeta |
|---------|-----------|------|
| Celulas vivas (Gen 100) | 776 | 1034 |
| Diferencia | - | **+33%** |

**Hallazgo clave**: La inicializacion estructurada produce mas celulas supervivientes, pero las correlaciones espaciales se pierden durante la evolucion B3/S23 clasica.

---

### Fase 2: Kernel de Vecindario Ponderado

**Archivo**: `zeta_gol_fase2.py`

**Concepto**: Reemplazar el kernel de Moore (3x3 uniforme) con un kernel ponderado por los ceros de zeta.

**Kernel Moore Clasico**:
```
[[1, 1, 1],
 [1, 0, 1],
 [1, 1, 1]]
```

**Kernel Zeta**:
```
K[x,y] = sum_gamma exp(-sigma * |gamma|) * cos(gamma * sqrt(x^2 + y^2))
```

**Modificaciones a las reglas**:
- En lugar de contar vecinos (entero 0-8)
- Calculamos suma ponderada (valor continuo)
- Umbrales ajustados: birth_range=(0.8, 1.5), survive_range=(0.3, 2.0)

**Resultados Fase 2**:
| Metrica | Moore Clasico | Zeta Ponderado |
|---------|---------------|----------------|
| Celulas vivas (Gen 100) | 776 | 1817 |
| Densidad | 0.078 | 0.182 |
| Diferencia | - | **+134%** |

**Hallazgo clave**: Las correlaciones espaciales ahora muestran oscilaciones caracteristicas de los ceros de zeta. El kernel preserva estructura durante la evolucion.

---

### Fase 3: Sistema Completo con Memoria Temporal

**Archivo**: `zeta_gol_fase3.py`

**Concepto**: Combinar todos los componentes para implementar el framework completo del paper:

1. **Inicializacion zeta** (Fase 1)
2. **Kernel espacial zeta** (Fase 2)
3. **Memoria temporal via L_zeros**
4. **Filtrado espectral**

**Ecuacion de evolucion**:
```
x(t+1) = GoL(x(t)) + alpha * Memory(history) + beta * Spectral(x(t))
```

**Componentes implementados**:

#### Operador de Laplace (ZetaLaplaceOperator)
```python
def apply_memory_filter(history):
    result = sum_tau K_sigma(tau) * history[t - tau]
    return result / total_weight
```

#### Filtro Espectral (ZetaSpectralFilter)
```python
H(omega) = sum_gamma exp(-sigma*|gamma|) / (1 + ((omega - gamma)/sigma)^2)
```
Funcion de transferencia con picos de resonancia en las frecuencias gamma_k.

**Resultados Fase 3**:
- Densidad final estable: ~0.18-0.20
- Autocorrelacion temporal sigue el kernel teorico
- Sistema robusto a variaciones de parametros alpha, beta

---

## 3. Hallazgos Principales

### 3.1 Efecto de Supervivencia
El kernel zeta incrementa significativamente la supervivencia de celulas:
- Fase 1: +33% vs aleatorio
- Fase 2: +134% vs Moore clasico

**Interpretacion**: Las correlaciones estructuradas de los ceros de zeta crean "nichos" favorables para la supervivencia.

### 3.2 Preservacion de Estructura
- **Sin kernel zeta**: Correlaciones se pierden rapidamente
- **Con kernel zeta**: Oscilaciones caracteristicas persisten

### 3.3 Memoria Temporal
La autocorrelacion temporal del sistema sigue el patron del kernel K_sigma(tau):
- Oscilaciones amortiguadas
- Decay lento (O(1/log|t|))
- Memoria de largo alcance

### 3.4 Espacio de Parametros
El sistema es robusto:
- alpha (memoria): 0.05 - 0.15 funcionan bien
- beta (espectral): 0.02 - 0.08 funcionan bien
- sigma (regularizacion): 0.05 - 0.1 optimo
- M (ceros): 20-30 suficiente para capturar estructura

---

## 4. Archivos Generados

### Codigo
| Archivo | Descripcion | Lineas |
|---------|-------------|--------|
| `zeta_game_of_life.py` | Fase 1: Inicializacion | ~530 |
| `zeta_gol_fase2.py` | Fase 2: Kernel ponderado | ~540 |
| `zeta_gol_fase3.py` | Fase 3: Sistema completo | ~670 |

### Visualizaciones
| Archivo | Descripcion |
|---------|-------------|
| `zeta_gol_inicial.png` | Estado inicial con ruido zeta |
| `zeta_kernel_2d.png` | Visualizacion del kernel en 2D |
| `zeta_correlations.png` | Funcion de correlacion espacial |
| `zeta_vs_random_comparison.png` | Comparacion Fase 1 |
| `zeta_vs_moore_comparison.png` | Comparacion Fase 2 |
| `zeta_parameter_sensitivity.png` | Sensibilidad a parametros |
| `zeta_full_system.png` | Sistema completo Fase 3 |
| `zeta_memory_comparison.png` | Efecto de memoria temporal |
| `zeta_alpha_beta_space.png` | Espacio de parametros |
| `zeta_full_system_highres.png` | Estado final alta resolucion |

---

## 5. Conexion con el Paper Original

| Seccion del Paper | Implementacion |
|-------------------|----------------|
| Kernel K_sigma(t) | `ZetaLaplaceOperator.kernel_temporal()` |
| Transformada L_zeros | `apply_memory_filter()` |
| Regularizacion Abel | Parametro sigma |
| Espacios de Schwartz | Discretizacion en grid finito |
| Operador Hilbert-Schmidt | Kernel con norma L2 finita |

---

## 6. Limitaciones Actuales

1. **Discretizacion**: El kernel continuo se discretiza en un grid finito
2. **Numero de ceros**: Solo usamos ~30 ceros (existen infinitos)
3. **Dimension**: Implementacion 2D (el paper menciona extensiones N-D)
4. **Complejidad**: O(M * N^2) por paso, donde M = ceros, N = tamanio grid

---

## 7. Proximos Pasos Sugeridos

1. **Extension a redes neuronales** (CNNs, Transformers)
2. **Kernels adaptativos** que evolucionan con el tiempo
3. **Analisis de patrones emergentes** (clasificacion automatica)
4. **Comparacion con otros autmatas** (Wireworld, Langton's Ant)
5. **Implementacion GPU** para escalabilidad
6. **Conexion con teoria de numeros** (correlaciones con distribucion de primos)

---

## 8. Como Ejecutar

```bash
# Fase 1
python zeta_game_of_life.py

# Fase 2
python zeta_gol_fase2.py

# Fase 3 (sistema completo)
python zeta_gol_fase3.py
```

**Dependencias**:
- numpy
- matplotlib
- scipy
- mpmath (opcional, para ceros exactos)

---

## 9. Evolucion: Zeta Neural Cellular Automata

### 9.1 Concepto

Combinacion de:
- **Neural Cellular Automata** (Google Distill): Patrones que crecen y se regeneran
- **Kernel Zeta**: Percepcion estructurada basada en ceros de Riemann

### 9.2 Arquitectura

```
Input: Estado (batch, 16 canales, H, W)
       |
       v
+------------------+
| Percepcion Zeta  |  <- Kernel zeta (no aprendible)
| + Filtros Sobel  |  <- Gradientes espaciales
+------------------+
       |
       v (batch, 64 canales, H, W)
+------------------+
| Update Network   |  <- MLP aprendible (Conv1x1)
| (hidden=96)      |
+------------------+
       |
       v
+------------------+
| Stochastic Mask  |  <- fire_rate=0.5
| + Alive Mask     |
+------------------+
       |
       v
Output: Estado actualizado
```

### 9.3 Resultados

| Metrica | Valor |
|---------|-------|
| Parametros totales | 7,776 |
| Loss inicial | ~0.20 |
| Loss final (200 iter) | ~0.06-0.08 |
| Dispositivo | CPU |

### 9.4 Observaciones

1. **Patrones diagonales**: El kernel zeta induce estructura diagonal caracteristica
2. **Crecimiento desde semilla**: El sistema aprende a expandirse desde un pixel
3. **Regeneracion parcial**: Intenta recuperar estructura despues de dano
4. **Necesita mas entrenamiento**: 2000+ iteraciones para resultados optimos

### 9.5 Archivo

`zeta_neural_ca.py` - Implementacion completa con PyTorch

---

## 10. Resumen de Archivos del Proyecto

### Codigo Principal
| Archivo | Descripcion | LOC |
|---------|-------------|-----|
| `zeta_game_of_life.py` | Fase 1: Inicializacion estructurada | ~530 |
| `zeta_gol_fase2.py` | Fase 2: Kernel ponderado | ~540 |
| `zeta_gol_fase3.py` | Fase 3: Sistema con memoria | ~670 |
| `zeta_neural_ca.py` | Evolucion: NCA diferenciable | ~590 |

### Documentacion
| Archivo | Descripcion |
|---------|-------------|
| `DOCUMENTACION.md` | Este documento |
| `EVOLUCIONES_PROPUESTAS.md` | Direcciones futuras |

### Visualizaciones Generadas
| Archivo | Fase |
|---------|------|
| `zeta_gol_inicial.png` | 1 |
| `zeta_kernel_2d.png` | 1 |
| `zeta_correlations.png` | 1 |
| `zeta_vs_random_comparison.png` | 1 |
| `zeta_kernel_fase2.png` | 2 |
| `zeta_vs_moore_comparison.png` | 2 |
| `zeta_parameter_sensitivity.png` | 2 |
| `zeta_gol_fase2_final.png` | 2 |
| `zeta_full_system.png` | 3 |
| `zeta_memory_comparison.png` | 3 |
| `zeta_alpha_beta_space.png` | 3 |
| `zeta_full_system_highres.png` | 3 |
| `zeta_nca_target.png` | NCA |
| `zeta_nca_growth.png` | NCA |
| `zeta_nca_regeneration.png` | NCA |
| `zeta_nca_loss.png` | NCA |

---

*Documentacion generada el 26 de Diciembre de 2025*
*Actualizada con resultados de Zeta Neural CA*
