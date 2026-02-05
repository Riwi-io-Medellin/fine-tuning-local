# Entrenamiento Paso a Paso

## Prerrequisitos

1. Docker instalado y funcionando
2. NVIDIA Container Toolkit (si tienes GPU)
3. El proyecto clonado y en el directorio

## Paso 1: Construir las imágenes

```bash
make build
```

Esto construye la imagen Docker con todas las dependencias:
- PyTorch con soporte CUDA
- Transformers, TRL, PEFT
- Unsloth (optimización)
- llama.cpp (conversión GGUF)

> **Primera vez**: Puede tardar 5-10 minutos en descargar y construir.

## Paso 2: Verificar el hardware

```bash
docker compose run --rm training python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

Deberías ver `GPU: True` si tu GPU es detectada.

## Paso 3: Revisar tu dataset

Verifica que tu CSV tenga el formato correcto:

```bash
head -n 5 data/sample_dataset.csv
```

Columnas esperadas: `instruction,response`

## Paso 4: Ejecutar el entrenamiento

```bash
make train
```

O con opciones personalizadas:

```bash
make train DATASET=mi_dataset.csv BASE_MODEL=unsloth/Llama-3.2-3B-Instruct
```

### Qué verás durante el entrenamiento

```
==============================================================
Fine-Tuning de LLM con LoRA/QLoRA
==============================================================
GPU detectada: NVIDIA GeForce RTX 3060 (12.0 GB)

Cargando modelo: unsloth/Llama-3.2-1B-Instruct
Modelo cargado con Unsloth (optimizado)
Parámetros entrenables: 3,407,872 / 1,235,814,400 (0.28%)

Cargando dataset: /app/data/sample_dataset.csv
Ejemplos de entrenamiento: 25

==============================================================
Iniciando entrenamiento...
==============================================================

{'loss': 2.1534, 'learning_rate': 0.0002, 'epoch': 0.4}
{'loss': 1.8923, 'learning_rate': 0.00019, 'epoch': 0.8}
...
```

### Interpretando los logs

| Métrica | Significado |
|---------|-------------|
| `loss` | Debe disminuir gradualmente |
| `learning_rate` | Decae según el scheduler |
| `epoch` | Progreso en el dataset |

**Señales de problemas:**
- Loss sube en lugar de bajar → learning rate muy alto
- Loss no baja después de varios epochs → datos con problemas
- Loss baja muy rápido y se estanca → posible overfitting

## Paso 5: Verificar el adaptador

Después del entrenamiento:

```bash
ls -la models/lora_adapter/
```

Archivos esperados:
- `adapter_config.json` - Configuración LoRA
- `adapter_model.safetensors` - Pesos del adaptador (~10-50MB)
- `tokenizer.json`, `tokenizer_config.json` - Tokenizador

## Hiperparámetros importantes

### Epochs

```python
num_train_epochs = 3  # Valor por defecto
```

- **1-2 epochs**: Datasets grandes (>5000 ejemplos)
- **3-5 epochs**: Datasets medianos (500-5000)
- **5-10 epochs**: Datasets pequeños (<500) - cuidado con overfitting

### Batch size y Gradient Accumulation

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
# Batch efectivo = 2 * 4 = 8
```

Si tienes poca memoria:
- Reduce `batch_size` a 1
- Aumenta `gradient_accumulation_steps`

### Learning Rate

```python
learning_rate = 2e-4  # Valor común para LoRA
```

- **1e-4 a 3e-4**: Rango típico para LoRA
- Más alto = aprendizaje más rápido pero menos estable
- Más bajo = más estable pero más lento

### Warmup

```python
warmup_ratio = 0.03  # 3% del total de pasos
```

Aumenta el learning rate gradualmente al inicio para estabilidad.

## Monitoreo avanzado

### Ver uso de GPU

En otra terminal:

```bash
watch -n 1 nvidia-smi
```

### Logs detallados

Modifica en `train.py`:

```python
logging_steps = 1  # Log cada paso (default: 10)
```

## Problemas comunes

### "CUDA out of memory"

**Soluciones en orden:**
1. Reduce `batch_size` a 1
2. Reduce `max_seq_length` a 1024
3. Usa un modelo más pequeño (1B en lugar de 3B)
4. Asegúrate de que no hay otros procesos usando la GPU

### "Model not found"

```bash
# Verificar conexión a HuggingFace
docker compose run --rm training python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### Entrenamiento muy lento en CPU

Es normal. Para datasets pequeños:
- Reduce epochs a 1-2
- Usa el modelo más pequeño posible
- Considera servicios de GPU en la nube para producción

## Próximo paso

Después del entrenamiento exitoso, continúa con:

```bash
make merge   # Fusionar adaptador con modelo base
```

Documentación: [05 - Conversión GGUF](05-conversion-gguf.md)
