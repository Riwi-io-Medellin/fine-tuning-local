# Conversión a GGUF

## ¿Por qué GGUF?

Después del entrenamiento tienes un modelo en formato HuggingFace (safetensors/pytorch). Para usarlo con Ollama, necesitas convertirlo a **GGUF**.

### ¿Qué es GGUF?

GGUF (GPT-Generated Unified Format) es el formato estándar para:
- **llama.cpp**: Inferencia eficiente en CPU/GPU
- **Ollama**: Gestión de modelos locales
- **LM Studio**: Interfaz gráfica para LLMs

### Ventajas

| Aspecto | HuggingFace | GGUF |
|---------|-------------|------|
| Inferencia | Requiere PyTorch + Transformers | Solo llama.cpp (~binario) |
| Tamaño | FP16 = 100% | Q4 = ~25% del tamaño |
| Velocidad | Buena con GPU | Excelente en CPU y GPU |
| Memoria | Alta | Baja con cuantización |

## El proceso de conversión

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│ Adaptador    │ ──▶ │ Modelo         │ ──▶ │ Modelo       │
│ LoRA         │     │ Fusionado FP16 │     │ GGUF Q4      │
│ (10-50MB)    │     │ (varios GB)    │     │ (1-2GB)      │
└──────────────┘     └────────────────┘     └──────────────┘
     make merge            make convert
```

## Paso 1: Fusionar LoRA con modelo base

```bash
make merge
```

Esto ejecuta:
1. Carga el modelo base (Llama-3.2-1B)
2. Carga los adaptadores LoRA entrenados
3. Fusiona los pesos (merge)
4. Guarda el resultado en `models/merged/`

### Verificar la fusión

```bash
ls -lh models/merged/
```

Deberías ver archivos `.safetensors` y configuración del modelo.

## Paso 2: Convertir a GGUF

```bash
make convert
```

O con opciones:

```bash
make convert QUANTIZATION=q5_k_m MODEL_NAME=mi-modelo
```

## Niveles de cuantización

| Nivel | Tamaño | Calidad | Uso recomendado |
|-------|--------|---------|-----------------|
| `f16` | 100% | Original | Máxima calidad, más memoria |
| `q8_0` | 50% | Muy buena | Cuando tienes memoria de sobra |
| `q6_k` | 40% | Buena | Balance calidad/tamaño |
| `q5_k_m` | 35% | Buena | Recomendado para calidad |
| `q4_k_m` | 25% | Aceptable | **Recomendado para tamaño** |
| `q4_0` | 22% | Baja | Mínimo viable |

### ¿Cuál elegir?

- **q4_k_m**: El estándar. Buena calidad con 75% de reducción de tamaño.
- **q5_k_m**: Si notas degradación con q4, usa este.
- **q8_0**: Si tienes memoria suficiente y quieres máxima calidad.

## Ejemplo de tamaños

Para un modelo Llama 3.2 1B:

| Formato | Tamaño aproximado |
|---------|-------------------|
| FP16 (original) | ~2.5GB |
| Q8_0 | ~1.3GB |
| Q5_K_M | ~900MB |
| Q4_K_M | ~700MB |

## Verificar el GGUF

```bash
ls -lh models/gguf/
```

Archivo esperado: `mi-modelo-finetuned.gguf`

## Proceso técnico detallado

### Conversión con llama.cpp

El script `convert_to_gguf.py` usa las herramientas de llama.cpp:

```bash
# 1. Convertir HF a GGUF FP16
python convert_hf_to_gguf.py modelo/ --outfile modelo_f16.gguf

# 2. Cuantizar
./llama-quantize modelo_f16.gguf modelo_q4.gguf Q4_K_M
```

### Conversión con Unsloth

Si Unsloth está disponible:

```python
model.save_pretrained_gguf(
    "output_dir",
    tokenizer,
    quantization_method="q4_k_m"
)
```

## Problemas comunes

### "llama.cpp not found"

El Dockerfile incluye llama.cpp, pero si falla:

```bash
docker compose run --rm training bash
cd /opt/llama.cpp
pip install -r requirements.txt
```

### "Conversion failed"

Verifica que el modelo fusionado esté completo:

```bash
ls models/merged/
# Debe incluir: config.json, *.safetensors, tokenizer*
```

### GGUF muy grande

Usa una cuantización más agresiva:

```bash
make convert QUANTIZATION=q4_0
```

## Próximo paso

Con el archivo GGUF listo, continúa con:

```bash
make deploy  # Crear modelo en Ollama
```

Documentación: [06 - Despliegue con Ollama](06-despliegue-ollama.md)
