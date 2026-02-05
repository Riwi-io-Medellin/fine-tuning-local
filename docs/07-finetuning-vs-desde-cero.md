# Fine-tuning vs Entrenar desde Cero

Este capítulo explica la diferencia real entre fine-tuning y pre-training, cuándo usar cada uno, y por qué el 99% de los casos no requieren entrenar desde cero.

## La Diferencia Real

### Fine-tuning (LoRA / QLoRA)

Partes de un modelo que **ya sabe "hablar"**.

El modelo ya:
- Entiende lenguaje
- Sabe razonar
- Tiene conocimiento general

Tú solo:
- Ajustas cómo responde
- Especializas un dominio
- Cambias estilo, formato o criterio

**Técnicamente:**
- No entrenas todos los pesos
- Entrenas adaptadores (LoRA)
- El modelo base queda congelado

**Analogía:** Un médico general al que le enseñas cardiología.

---

### Entrenar desde Cero (Pretraining)

El modelo **no sabe absolutamente nada**.

- No entiende palabras
- No sabe gramática
- No razona
- Solo ve números al inicio

**Técnicamente:**
- Inicializas pesos aleatorios
- Entrenas TODAS las capas
- Aprendes: tokens, sintaxis, semántica, razonamiento

**Analogía:** Enseñar a leer a un recién nacido con Wikipedia entera.

---

## Comparativa Técnica

| Aspecto | Fine-tuning | Desde cero |
|---------|------------|------------|
| Pesos entrenados | Solo LoRA (~1-5%) | Todos (100%) |
| Dataset | Miles de ejemplos | Billones de tokens |
| Costo | Bajo | Altísimo |
| Tiempo | Horas | Semanas / meses |
| Hardware | 1 GPU | Cluster |
| Riesgo | Bajo | Muy alto |
| Objetivo | Especializar | Crear base |

---

## Qué Debes Entender ANTES de Entrenar desde Cero

Entrenar desde cero no es solo "más epochs". Es otro mundo.

### 1. Tokenización (CRÍTICO)

Tienes que decidir:
- BPE, SentencePiece, Unigram
- Tamaño del vocabulario
- Idiomas soportados

> Un tokenizer malo = modelo malo, aunque entrenes meses.

### 2. Arquitectura del Modelo

Decisiones irreversibles:
- Número de capas
- Dimensión del embedding
- Atención (Multi-head, GQA, MQA)
- Positional encoding (RoPE, ALiBi)

**Ejemplo real:** Cambiar GQA afecta memoria, velocidad y calidad.

### 3. Objetivo de Entrenamiento

Normalmente:
- Causal Language Modeling
- Predecir el siguiente token

También defines:
- Context window
- Masking
- Curriculum learning

### 4. Dataset Masivo

Para que un modelo sea usable:
- 100B – 1T tokens
- Limpieza brutal
- Deduplicación
- Mezcla de dominios: código, texto, conversaciones, documentación

> Más datos > más epochs.

### 5. Estabilidad Numérica

Aquí mueren muchos intentos:
- Learning rate schedule
- Warmup
- Gradient clipping
- Mixed precision
- Overflow / NaNs

Fine-tuning no te prepara para esto.

---

## Hardware Real para Entrenar desde Cero

Para poner los pies en la tierra:

| Modelo | GPUs necesarias |
|--------|-----------------|
| 1B | 8–16 A100 |
| 7B | 64–128 A100 |
| 13B | 256+ A100 |
| 70B | Datacenter completo |

**Costos:** Cientos de miles o millones de dólares.

---

## Cuándo Tiene Sentido Entrenar desde Cero

### SÍ tiene sentido si:
- Nuevo idioma subrepresentado
- Nuevo alfabeto / dominio radical
- Investigación académica
- Infraestructura propia
- Objetivo fundacional (startup / lab)

### NO tiene sentido si:
- Quieres un asistente
- Quieres un experto en X
- Quieres estilo propio
- Quieres enseñar

> En 2026, el 99% de los casos no deben entrenar desde cero.

---

## El Punto Clave

Esto es lo que debes entender:

> **Fine-tuning no enseña conocimiento nuevo, enseña cómo usar el conocimiento existente.**

**Ejemplo:**
- El modelo no "aprende Rails" con fine-tuning
- Aprende **cómo responder** como experto en Rails
- El conocimiento de Rails ya estaba en el modelo base

---

## Continued Pretraining (Punto Medio)

Si quieres ir un nivel más arriba sin la locura de entrenar desde cero:

### Qué es
- Tomas un modelo base
- Entrenas con mucho texto de tu dominio
- Sigues entrenando todos los pesos (no solo LoRA)

### Cómo hacerlo
1. Modelo base existente
2. 50–100M tokens del dominio específico
3. Learning rate muy bajo

Es el punto medio: más potente que fine-tuning, menos costoso que desde cero.

---

## Resumen

| Concepto | Descripción |
|----------|-------------|
| Fine-tuning | Especializar un modelo existente |
| Desde cero | Crear un modelo que entienda lenguaje |
| Hardware | Define la estrategia posible |
| Dataset | Manda más que el modelo |
| Desde cero | Es un **proyecto**, no un tutorial |

### En este Repositorio

Este repositorio se enfoca en **fine-tuning con LoRA/QLoRA** porque:
- Es reproducible en hardware accesible
- Resuelve el 99% de los casos de uso reales
- Se puede ejecutar en una sola GPU o incluso CPU
- El tiempo de iteración es de horas, no semanas

Para la mayoría de aplicaciones (asistentes, expertos de dominio, estilos personalizados), fine-tuning es la respuesta correcta.
