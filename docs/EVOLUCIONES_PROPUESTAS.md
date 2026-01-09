# Evoluciones Propuestas para el Sistema Zeta GoL

## Investigacion de Estado del Arte (Diciembre 2025)

Basado en investigacion de literatura reciente, identificamos las siguientes direcciones para evolucionar el sistema.

---

## 1. Lenia Zeta: Automata Celular Continuo

### Concepto
Lenia es una generalizacion continua del Game of Life con estados, espacio y tiempo continuos. La conexion con nuestro kernel zeta es natural.

### Modificacion Propuesta
Reemplazar el kernel Gaussiano de Lenia con nuestro kernel zeta:

**Lenia clasico**:
```
K(r) = exp(-r^2 / (2*sigma^2))  # Kernel Gaussiano
G(u) = 2*exp(-((u-mu)/sigma)^2) - 1  # Funcion de crecimiento
```

**Lenia Zeta**:
```
K_zeta(r) = sum_gamma exp(-sigma*|gamma|) * cos(gamma * r)  # Kernel Zeta
G_zeta(u, t) = funcion modulada por ceros de zeta
```

### Beneficios Esperados
- Patrones mas complejos ("criaturas" con memoria)
- Correlaciones de largo alcance
- Posible emergencia de comportamiento mas rico

### Complejidad
Media - Requiere adaptar framework Lenia existente

---

## 2. DiffLogic Zeta CA: Automata Logico Diferenciable

### Concepto
Google introdujo Differentiable Logic Cellular Automata (DiffLogic CA) que combina:
- Compuertas logicas diferenciables
- Neural Cellular Automata (NCA)
- Entrenamiento end-to-end

### Modificacion Propuesta
Incorporar los ceros de zeta en la inicializacion de las compuertas logicas:

```
W_init = ZetaKernel(shape)  # Pesos iniciales estructurados
Logic_gates = softmax(W_init * temperature)  # Compuertas diferenciables
```

### Beneficios Esperados
- Descubrimiento automatico de reglas
- Inferencia discreta (eficiente en hardware)
- Patrones regenerativos

### Complejidad
Alta - Requiere implementar framework DiffLogic

---

## 3. ViTCA Zeta: Vision Transformer Cellular Automata

### Concepto
Attention-based Neural Cellular Automata usa self-attention localizado para lograr organizacion global a traves de interacciones locales.

### Modificacion Propuesta
Usar el kernel zeta como bias posicional en el mecanismo de atencion:

**Atencion clasica**:
```
Attention(Q, K, V) = softmax(Q*K^T / sqrt(d)) * V
```

**Atencion Zeta**:
```
Attention_zeta(Q, K, V) = softmax(Q*K^T / sqrt(d) + ZetaBias(pos)) * V
```

donde `ZetaBias(pos)` codifica las correlaciones zeta en las posiciones.

### Beneficios Esperados
- Auto-organizacion global desde interacciones locales
- Complejidad lineal amortizada en tiempo
- Conexion con transformers modernos

### Complejidad
Alta - Requiere implementar ViTCA + modificaciones

---

## 4. Flow Lenia Zeta: Multi-Especies con Parametros Localizados

### Concepto
Flow Lenia extiende Lenia con:
- Conservacion de masa
- Parametros localizados (cada "criatura" tiene su propio kernel)
- Evolucion abierta (open-ended)

### Modificacion Propuesta
```
K_local(r, x, y) = sum_gamma w_gamma(x,y) * exp(-sigma*|gamma|) * cos(gamma * r)
```

Los pesos `w_gamma(x,y)` varian espacialmente, permitiendo diferentes "especies" con diferentes configuraciones de ceros de zeta.

### Beneficios Esperados
- Ecosistemas artificiales con memoria
- Evolucion de parametros
- Diversidad de comportamientos

### Complejidad
Alta - Extension significativa del sistema actual

---

## 5. Zeta Neural CA para Regeneracion

### Concepto
Growing Neural Cellular Automata (Google Distill) aprende patrones que:
- Crecen desde una semilla
- Se regeneran ante dano
- Son robustos a perturbaciones

### Modificacion Propuesta
Usar el kernel zeta como capa de convolucion:

```python
class ZetaNCA(nn.Module):
    def __init__(self):
        self.zeta_conv = ZetaConv2D(in_channels, out_channels, M=30, sigma=0.1)
        self.update_net = nn.Sequential(...)

    def forward(self, x):
        perception = self.zeta_conv(x)  # Percepcion con kernel zeta
        update = self.update_net(perception)
        return x + update * alive_mask
```

### Beneficios Esperados
- Regeneracion con memoria de largo alcance
- Robustez mejorada
- Patrones mas organicos

### Complejidad
Media - Modificacion directa de arquitectura NCA

---

## 6. Hibrido: Zeta-Transformer para Secuencias de CA

### Concepto
Usar un Transformer para predecir la evolucion del CA zeta, aprovechando que los transformers pueden aprender dinamicas estocasticas.

### Arquitectura Propuesta
```
Input: Historia de estados [x(t-k), ..., x(t)]
       ↓
Zeta Embedding (codifica con kernel zeta)
       ↓
Transformer Encoder (self-attention)
       ↓
Zeta Decoder (genera siguiente estado)
       ↓
Output: x(t+1)
```

### Beneficios Esperados
- Prediccion de largo plazo
- Aprendizaje de patrones emergentes
- Generalizacion a nuevas condiciones iniciales

### Complejidad
Media-Alta - Combina arquitecturas conocidas

---

## Recomendacion de Prioridad

### Fase Inmediata (Baja Complejidad)
1. **Zeta Neural CA para Regeneracion** - Extension natural del trabajo actual
2. **Lenia Zeta** - Framework bien documentado

### Fase Media (Complejidad Media)
3. **Hibrido Zeta-Transformer** - Combina con ML moderno
4. **Flow Lenia Zeta** - Ecosistemas artificiales

### Fase Avanzada (Alta Complejidad)
5. **ViTCA Zeta** - Atencion + CA + Zeta
6. **DiffLogic Zeta** - Frontera de investigacion

---

## Recursos y Referencias

1. [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
2. [Differentiable Logic CA (Google)](https://google-research.github.io/self-organising-systems/difflogic-ca/)
3. [Lenia Project](https://chakazul.github.io/lenia.html)
4. [Flow Lenia](https://sites.google.com/view/flowlenia/)
5. [Attention-based NCA (NeurIPS 2022)](https://arxiv.org/abs/2211.01233)
6. [Learning Sensorimotor Agency in CA](https://developmentalsystems.org/sensorimotor-lenia/)

---

## Siguiente Paso Sugerido

Implementar **Opcion 5: Zeta Neural CA para Regeneracion** porque:
1. Es la extension mas natural del trabajo actual
2. Usa frameworks existentes (PyTorch)
3. Demuestra claramente el valor del kernel zeta
4. Conecta directamente con el paper original (robustez, memoria)

---

*Documento generado el 26 de Diciembre de 2025*
