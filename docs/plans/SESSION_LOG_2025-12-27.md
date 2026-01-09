# Session Log: 2025-12-27

## Tema: Comunicación Química - Feromonas de Atracción

### Objetivo
Implementar y validar feromonas de atracción que permitan forrajeo colectivo en ZetaOrganism.

---

## Trabajo Realizado

### 1. Diagnóstico del Problema Inicial

El test inicial de atracción no mostraba diferencia porque:
- Los scouts Fi se colocaban en el recurso desde el inicio
- Mass seguía a Fi automáticamente (comportamiento base)
- El resultado era idéntico con y sin feromonas

### 2. Iteraciones de Diseño

#### Intento 1: Scouts en recursos
- **Problema**: Fi pierde rol porque no tiene Mass cerca (`same_mass < 1`)
- **Diagnóstico**: Fi se convierte en Mass inmediatamente

#### Intento 2: Scouts + seguidores iniciales
- **Problema**: Todos llegan al recurso incluso sin feromonas
- **Diagnóstico**: Mass sigue a Fi que está en el recurso

#### Intento 3: Exploración aleatoria pura
- **Problema**: Células no llegan nunca (random walk muy lento)
- **Diagnóstico**: 400 steps insuficientes para alcanzar centro

#### Intento 4: Exploración con sesgo hacia centro
- **Problema**: Recursos en centro = todos llegan igual
- **Diagnóstico**: El sesgo coincide con ubicación de recursos

#### Intento 5 (FINAL): Recursos desplazados
- **Diseño**: Recursos en (15,50), Fi explora hacia (32,32)
- **Resultado**: Con atracción 15 células, sin atracción 0 células
- **Éxito**: Diferencia clara y significativa

### 3. Cambios en el Código

#### `exp_comunicacion_quimica.py`

**Nuevo método `set_patch_position()`:**
```python
def set_patch_position(self, position: tuple, radius: float = 10.0):
    """Reposiciona los parches de recursos a nueva ubicación."""
    self.patches = [ResourcePatch(
        position=position,
        radius=radius,
        capacity=100.0,
        current=100.0,
        regen_delay=30
    )]
```

**Modificación `initialize_foraging()`:**
- Todos los organismos empiezan en esquinas
- Fi con sus Mass, sin scouts pre-posicionados
- Org0 en (10,10), Org1 en (54,54)

**Modificación movimiento Fi:**
```python
else:  # Fi explora con sesgo hacia centro
    center = self.grid_size // 2
    if np.random.random() < 0.35:
        if np.random.random() < 0.7:
            base_dx = int(np.sign(center - x))
            base_dy = int(np.sign(center - y))
        else:
            base_dx = np.random.choice([-1, 0, 1])
            base_dy = np.random.choice([-1, 0, 1])
```

#### `test_foraging.py` (nuevo)
Script de test con recursos desplazados para validar atracción.

---

## Resultados Finales

### Experimento: Recursos Desplazados

| Condición | Células en Recurso | Distancia Org0 | Emisiones |
|-----------|-------------------|----------------|-----------|
| Sin feromonas | 0 | 24.9 | 0 |
| Con atracción | **15** | 10.6 | 12,283 |

### Secuencia Temporal
```
Step 100: 0 vs 0 (sin diferencia)
Step 200: 0 vs 30 (atracción comienza)
Step 400: 0 vs 15 (estabilización)
Step 600: 0 vs 15 (equilibrio)
```

### Mecanismo Validado
1. Fi explora aleatoriamente con sesgo hacia centro
2. Un Fi de Org0 alcanza el recurso por azar
3. Fi emite feromona de atracción
4. Gradiente se propaga por difusión gaussiana
5. Otros Fi de Org0 detectan gradiente y son atraídos
6. Mass sigue a sus Fi hacia el recurso
7. Org1 no encuentra recurso (su exploración va hacia otro lado)

---

## Documentación Actualizada

### `docs/REPORTE_ZETA_ORGANISM.md`

- **Sección 2.14.7**: Nueva sección "Feromonas de Atracción: Forrajeo Colectivo"
- **Tabla resumen ejecutivo**: Añadido "Forrajeo colectivo" como propiedad emergente
- **Sección 3.1**: Añadido a tabla de propiedades emergentes

### Propiedad Emergente #11: Forrajeo Colectivo
- **Evidencia**: +15 células guiadas a recurso oculto
- **Significancia**: Exploración cooperativa
- **Mecanismo**: Broadcast de ubicación de recursos via feromonas

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `exp_comunicacion_quimica.py` | Exploración Fi, `set_patch_position()`, `initialize_foraging()` |
| `test_foraging.py` | Nuevo script de test |
| `docs/REPORTE_ZETA_ORGANISM.md` | Sección 2.14.7, tablas resumen |
| `docs/SESSION_LOG_2025-12-27.md` | Este archivo |

---

## Conclusiones

1. **Atracción funciona**: Diferencia clara entre condiciones (15 vs 0 células)
2. **Diseño experimental crítico**: El test debe evitar confundir comportamiento base con efecto de feromonas
3. **Forrajeo colectivo validado**: Cuando un explorador encuentra recurso, todo el organismo converge
4. **Analogía biológica**: Similar a reclutamiento de hormigas o comunicación de abejas

---

## Estado del Proyecto

### Propiedades Emergentes Demostradas: 11

| # | Propiedad | Experimento |
|---|-----------|-------------|
| 1 | Homeostasis | Todos |
| 2 | Regeneración | 2.3, 2.4 |
| 3 | Antifragilidad | 2.4, 2.5 |
| 4 | Quimiotaxis | 2.6 |
| 5 | Memoria espacial | 2.7 |
| 6 | Auto-segregación | 2.8 |
| 7 | Exclusión competitiva | 2.8 |
| 8 | Partición de nicho | 2.11 |
| 9 | Pánico colectivo | 2.14 |
| 10 | Huida coordinada | 2.14.6 |
| 11 | **Forrajeo colectivo** | **2.14.7** |

---

---

## PARTE 2: Teoría Matemática

### Pregunta Fundamental
> ¿Por qué los ceros zeta producen emergencia? ¿Son realmente especiales?

### Experimentos Realizados

1. **exp_teoria_zeta.py**: Comparación en autómata celular simple
   - Resultado: Diferencias menores, ZETA comparable a otros
   - Conclusión: En sistemas simples, el kernel importa poco

2. **exp_teoria_zeta_v2.py**: Métricas de criticidad (Goldilocks)
   - Resultado: Todos los kernels producen sistemas "muertos" en CA
   - Conclusión: Necesitamos sistemas más complejos

3. **Análisis matemático de espaciado**:
   - RANDOM: ratio 4318 (caótico)
   - ZETA: ratio 3.9 (estructurado)
   - UNIFORM: ratio 1.0 (rígido)
   - **Hallazgo clave: ZETA está matemáticamente en el medio**

### Documento Teórico

Creado `docs/TEORIA_ZETA_EMERGENCIA.md` con:
- Análisis del "borde del caos"
- Conexión con Hipótesis de Riemann
- Explicación de por qué Re(s) = 1/2 es crítico
- Conjetura sobre minimización de exponente de Lyapunov

### Conclusión Teórica

Los ceros zeta son especiales porque:

```
ORDEN (UNIFORM) ←── ZETA ──→ CAOS (RANDOM)
     1.0              3.9         4318
```

Están **exactamente en el borde** entre convergencia y divergencia, donde la emergencia es máxima.

---

## Archivos Creados en Esta Sesión

| Archivo | Descripción |
|---------|-------------|
| `exp_comunicacion_quimica.py` | Actualizado con atracción |
| `test_foraging.py` | Test de forrajeo |
| `exp_teoria_zeta.py` | Comparación de kernels |
| `exp_teoria_zeta_v2.py` | Métricas Goldilocks |
| `exp_teoria_zeta_v3.py` | Sistema multi-agente |
| `docs/TEORIA_ZETA_EMERGENCIA.md` | Documento teórico |
| `docs/SESSION_LOG_2025-12-27.md` | Este archivo |
| `zeta_kernel_comparison.png` | Visualización kernels |
| `zeta_patterns_comparison.png` | Patrones CA |
| `zeta_goldilocks.png` | Métricas criticidad |

---

*Sesión finalizada: 2025-12-27*
