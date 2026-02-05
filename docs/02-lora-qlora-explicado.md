# LoRA y QLoRA Explicados

## El problema: Fine-tuning tradicional es caro

Un LLM de 7B parámetros en FP16 ocupa **14GB** solo para los pesos. Para entrenar, necesitas además:
- Gradientes (~14GB más)
- Estados del optimizador (~28GB más)

**Total: ~56GB de VRAM**. Inviable para hardware de consumo.

## La solución: LoRA (Low-Rank Adaptation)

### Idea central

En lugar de modificar los **14 billones de parámetros** del modelo, LoRA:

1. **Congela** el modelo original (no se entrena)
2. **Agrega** matrices pequeñas que sí se entrenan
3. **Combina** ambos en inferencia

### Visualmente

```
                    ┌─────────────────┐
                    │   Modelo Base   │
Entrada ──────────▶ │  (CONGELADO)    │ ──────────┐
                    └─────────────────┘           │
                                                  │ Suma
                    ┌─────────────────┐           │
                    │  Adaptador LoRA │           ▼
          └───────▶ │   (ENTRENA)     │ ──────▶ Salida
                    │   (Pequeño)     │
                    └─────────────────┘
```

### Las matemáticas (simplificadas)

Una capa normal hace:

```
y = Wx  (donde W es una matriz enorme)
```

LoRA modifica esto a:

```
y = Wx + BAx  (donde B y A son matrices pequeñas)
```

- **W**: Matriz original (congelada), digamos 4096 x 4096 = 16M parámetros
- **B**: Matriz pequeña 4096 x 16 = 65K parámetros  
- **A**: Matriz pequeña 16 x 4096 = 65K parámetros
- **BA**: El producto reconstruye una matriz del tamaño de W

**El truco**: El "rank" (16 en este ejemplo) determina la capacidad del adaptador. Ranks bajos = menos parámetros pero menos expresividad.

### Ventajas de LoRA

| Aspecto | Fine-tuning tradicional | LoRA |
|---------|------------------------|------|
| Parámetros entrenados | 100% (~7B) | ~0.1% (~7M) |
| Memoria GPU | 40-60GB | 8-16GB |
| Tiempo de entrenamiento | Horas/días | Minutos/horas |
| Tamaño del modelo guardado | 14GB | ~10-100MB |
| Cambiar de tarea | Cargar modelo completo | Cargar adapter pequeño |

## QLoRA: LoRA + Cuantización

### Cuantización del modelo base

QLoRA va un paso más allá: **carga el modelo base en 4-bit**.

```
Modelo 7B en FP16:  14GB
Modelo 7B en 4-bit:  3.5GB  (75% menos)
```

### Cómo funciona

1. El modelo base se carga cuantizado (NF4)
2. Los adaptadores LoRA se entrenan en FP16/BF16
3. Los gradientes fluyen a través del modelo cuantizado

### Memoria estimada para entrenar

| Modelo | FP16 completo | LoRA FP16 | QLoRA 4-bit |
|--------|---------------|-----------|-------------|
| 3B | ~24GB | ~12GB | ~6GB |
| 7B | ~56GB | ~24GB | ~12GB |
| 13B | ~104GB | ~40GB | ~20GB |

**Conclusión**: QLoRA hace posible fine-tunear modelos de 7B en una RTX 3060 de 12GB.

## Parámetros importantes

### Rank (r)

El tamaño de las matrices LoRA.

- **r=8**: Muy eficiente, menos capacidad
- **r=16**: Balance común (recomendado para empezar)
- **r=32**: Más capacidad, más memoria
- **r=64**: Para tareas complejas

### Alpha

Factor de escala para los adaptadores.

- Generalmente se usa `alpha = rank`
- Valores más altos = adaptaciones más fuertes
- Fórmula real: `scaling = alpha / rank`

### Target modules

Qué capas modificar. Para modelos Llama:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Atención
    "gate_proj", "up_proj", "down_proj",      # Feed-forward
]
```

Entrenar todas estas capas da más capacidad de adaptación.

### Dropout

Regularización para evitar overfitting.

- **0**: Sin dropout (común en Unsloth)
- **0.05-0.1**: Algo de regularización
- **0.2+**: Mucha regularización (datasets pequeños)

## Código ejemplo

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=16,           # Alpha (generalmente = rank)
    target_modules=[         # Qué capas modificar
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # Dropout para regularización
    bias="none",             # No entrenar bias
    task_type="CAUSAL_LM"    # Tipo de tarea
)
```

## Visualización de memoria

```
┌────────────────────────────────────────────────────┐
│                   GPU VRAM (12GB)                   │
├────────────────────────────────────────────────────┤
│ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Modelo 7B 4-bit (3.5GB)                             │
├────────────────────────────────────────────────────┤
│ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Adaptadores LoRA (0.1GB)                            │
├────────────────────────────────────────────────────┤
│ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Gradientes + Optimizador (~4GB)                     │
├────────────────────────────────────────────────────┤
│ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Batch + Activaciones (~2GB)                         │
├────────────────────────────────────────────────────┤
│                   Libre (~2GB)                      │
└────────────────────────────────────────────────────┘
Total: ~9.5GB → ¡Cabe en 12GB!
```

## Siguiente paso

Continúa con [03 - Preparación de Datos](03-preparacion-datos.md) para aprender a estructurar tu dataset.
