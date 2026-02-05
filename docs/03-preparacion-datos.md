# Preparación de Datos

## La regla de oro

> El dataset importa más que el modelo, el algoritmo y los hiperparámetros combinados.

Un dataset pequeño pero limpio produce mejores resultados que uno grande pero ruidoso.

## Formato básico: Instruction - Response

Para modelos de tipo "Instruct", el formato estándar es:

```csv
instruction,response
"Tu pregunta o instrucción aquí","La respuesta ideal aquí"
```

### Ejemplo práctico

```csv
instruction,response
"¿Qué es un callback en Rails?","Un callback en Rails es un método que se ejecuta automáticamente en ciertos momentos del ciclo de vida de un objeto ActiveRecord. Por ejemplo, before_save se ejecuta antes de guardar, after_create después de crear. Son útiles para lógica que siempre debe ocurrir, como generar un slug o enviar una notificación."
"Explica qué hace belongs_to en Rails","belongs_to establece una relación donde el modelo actual tiene una clave foránea que referencia a otro modelo. Por ejemplo, si un Comment belongs_to :post, entonces la tabla comments tiene una columna post_id. Rails automáticamente crea métodos como comment.post para acceder al Post asociado."
```

## Estructura de chat (para conversaciones)

Si tu modelo necesita manejar contexto de conversación:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un asistente experto en Rails"},
    {"role": "user", "content": "¿Qué es ActiveRecord?"},
    {"role": "assistant", "content": "ActiveRecord es el ORM de Rails..."},
    {"role": "user", "content": "¿Y cómo hago una migración?"},
    {"role": "assistant", "content": "Para crear una migración..."}
  ]
}
```

## Cantidad de datos recomendada

| Objetivo | Cantidad |
|----------|----------|
| Cambio de estilo/formato | 100-500 ejemplos |
| Tarea específica nueva | 500-2000 ejemplos |
| Dominio técnico profundo | 2000-5000 ejemplos |
| Comportamiento muy diferente | 5000+ ejemplos |

> **Regla práctica**: Empieza con 200-500 ejemplos de alta calidad. Aumenta solo si ves que el modelo no generaliza.

## Características de un buen dataset

### 1. Consistencia en estilo

**Malo:**
```
- Respuesta 1: "Es simple: haz X"
- Respuesta 2: "Para realizar esta operación, deberá seguir los siguientes pasos detallados..."
- Respuesta 3: "x hace y"
```

**Bueno:**
```
- Respuesta 1: "Para hacer X, sigue estos pasos: 1) ... 2) ..."
- Respuesta 2: "Para lograr Y, el proceso es: 1) ... 2) ..."
- Respuesta 3: "Para implementar Z, necesitas: 1) ... 2) ..."
```

### 2. Respuestas completas pero concisas

**Malo:**
```
response: "Sí"
response: "Usa Rails"
```

**Bueno:**
```
response: "Sí, puedes usar concerns en Rails para compartir código entre modelos. Crea un archivo en app/models/concerns y usa 'extend ActiveSupport::Concern'."
```

### 3. Diversidad en las instrucciones

**Malo:** 100 variaciones de "¿Qué es X?"

**Bueno:** Mezcla de:
- Preguntas directas
- Solicitudes de código
- Pedidos de explicación
- Corrección de errores
- Comparaciones

### 4. Sin contradicciones

Evita ejemplos que digan cosas opuestas para el mismo tipo de pregunta.

## Limpieza de datos

### Checklist antes de entrenar

- [ ] ¿Todas las respuestas están completas?
- [ ] ¿El formato es consistente (CSV válido, JSON válido)?
- [ ] ¿Hay duplicados que deban eliminarse?
- [ ] ¿Las respuestas tienen el estilo deseado?
- [ ] ¿La información es correcta y actualizada?
- [ ] ¿Se removió información personal/sensible?

### Script de validación simple

```python
import pandas as pd

df = pd.read_csv('tu_dataset.csv')

# Verificar columnas
assert 'instruction' in df.columns, "Falta columna 'instruction'"
assert 'response' in df.columns, "Falta columna 'response'"

# Verificar valores vacíos
print(f"Instructions vacías: {df['instruction'].isna().sum()}")
print(f"Responses vacías: {df['response'].isna().sum()}")

# Longitudes
print(f"\nLongitud promedio instruction: {df['instruction'].str.len().mean():.0f}")
print(f"Longitud promedio response: {df['response'].str.len().mean():.0f}")

# Duplicados
print(f"\nDuplicados: {df.duplicated().sum()}")

# Muestra
print("\nEjemplo:")
print(df.iloc[0])
```

## Fuentes de datos

### Datos propios
- Logs de soporte técnico (anonimizados)
- Documentación interna
- FAQs existentes
- Transcripciones de capacitaciones
- Casos reales de clientes

### Datos sintéticos
Puedes usar un LLM más grande para generar datos de entrenamiento:

```python
prompt = """Genera 10 pares de pregunta-respuesta sobre {tema}.
Formato JSON: [{"instruction": "...", "response": "..."}]
Las respuestas deben ser técnicas pero accesibles."""
```

### Datasets públicos (para practicar)
- Alpaca dataset
- OpenAssistant conversations
- Datasets de HuggingFace Hub

## Errores comunes

| Error | Consecuencia |
|-------|--------------|
| Respuestas muy cortas | Modelo da respuestas incompletas |
| Respuestas muy largas | Modelo divaga sin foco |
| Datos sucios/ruidosos | Modelo aprende patrones erróneos |
| Poca diversidad | Modelo solo funciona para casos específicos |
| Información incorrecta | Modelo genera respuestas falsas |

## Ejemplo de dataset para este tutorial

El archivo `data/sample_dataset.csv` incluye ~25 ejemplos sobre fine-tuning de LLMs. Puedes usarlo para probar el flujo completo y después reemplazarlo con tus propios datos.

## Siguiente paso

Continúa con [04 - Entrenamiento paso a paso](04-entrenamiento-paso-a-paso.md) para ejecutar el entrenamiento.
