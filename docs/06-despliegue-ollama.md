# Despliegue con Ollama

## ¿Qué es Ollama?

Ollama es una herramienta para ejecutar LLMs localmente. Simplifica:
- Gestión de modelos
- Configuración de prompts
- API REST para integración
- Inferencia eficiente

## Paso 1: Iniciar Ollama

```bash
make ollama-up
```

Verifica que esté funcionando:

```bash
curl http://localhost:11434/api/version
```

## Paso 2: Crear el modelo

```bash
make deploy MODEL_NAME=mi-asistente
```

Esto ejecuta:
1. Genera un Modelfile completo
2. Ejecuta `ollama create` con el GGUF
3. El modelo queda disponible en Ollama

## El Modelfile explicado

El Modelfile es la configuración del modelo:

```dockerfile
# Archivo GGUF base
FROM /models/mi-modelo-finetuned.gguf

# Template de prompt (formato Llama 3)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# Mensaje de sistema por defecto
SYSTEM """Eres un asistente técnico experto y útil."""

# Parámetros de generación
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

### Parámetros importantes

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `temperature` | Creatividad/aleatoriedad | 0.1-1.0 (0.7 default) |
| `top_p` | Nucleus sampling | 0.9-0.95 |
| `num_ctx` | Tamaño del contexto | 2048-8192 |
| `num_predict` | Máximo tokens a generar | -1 (sin límite) |
| `repeat_penalty` | Penalización de repetición | 1.0-1.2 |

## Paso 3: Probar el modelo

```bash
make chat
```

O directamente:

```bash
docker compose exec ollama ollama run mi-asistente
```

### Interacción

```
>>> ¿Qué es fine-tuning?
Fine-tuning es el proceso de especializar un modelo de lenguaje pre-entrenado
para una tarea o dominio específico...

>>> /bye
```

## API REST de Ollama

Ollama expone una API en `http://localhost:11434`:

### Generar texto

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "mi-asistente",
  "prompt": "¿Qué es LoRA?",
  "stream": false
}'
```

### Formato chat

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "mi-asistente",
  "messages": [
    {"role": "user", "content": "Hola, ¿cómo funciona QLoRA?"}
  ]
}'
```

### Listar modelos

```bash
curl http://localhost:11434/api/tags
```

## Integración con aplicaciones

### Python

```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'mi-asistente',
    'prompt': '¿Qué es fine-tuning?',
    'stream': False
})

print(response.json()['response'])
```

### JavaScript/Node

```javascript
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  body: JSON.stringify({
    model: 'mi-asistente',
    prompt: '¿Qué es fine-tuning?',
    stream: false
  })
});

const data = await response.json();
console.log(data.response);
```

### Ruby/Rails

```ruby
require 'net/http'
require 'json'

uri = URI('http://localhost:11434/api/generate')
response = Net::HTTP.post(uri, {
  model: 'mi-asistente',
  prompt: '¿Qué es fine-tuning?',
  stream: false
}.to_json)

puts JSON.parse(response.body)['response']
```

## Gestión de modelos

### Listar modelos instalados

```bash
docker compose exec ollama ollama list
```

### Eliminar un modelo

```bash
docker compose exec ollama ollama rm mi-asistente
```

### Información del modelo

```bash
docker compose exec ollama ollama show mi-asistente
```

## Ollama nativo (sin Docker)

Si prefieres usar Ollama instalado en tu sistema:

1. Instala Ollama: https://ollama.com/download

2. Crea el modelo directamente:

```bash
# Copia el GGUF a una ubicación accesible
cp models/gguf/mi-modelo-finetuned.gguf ~/.ollama/

# Crea Modelfile
cat > Modelfile << EOF
FROM ~/.ollama/mi-modelo-finetuned.gguf
# ... resto de configuración
EOF

# Crea el modelo
ollama create mi-asistente -f Modelfile
```

## Comparar antes/después

Script para comparar el modelo base vs fine-tuned:

```bash
# Modelo base
ollama run llama3.2:1b "¿Qué es fine-tuning?"

# Modelo fine-tuned
ollama run mi-asistente "¿Qué es fine-tuning?"
```

## Problemas comunes

### "Model not found"

```bash
# Verificar que el modelo se creó
docker compose exec ollama ollama list

# Si no aparece, recrear
make deploy
```

### "GGUF file not found"

Verifica que el volumen esté montado correctamente:

```bash
docker compose exec ollama ls /models/
```

### Respuestas lentas

- Usa cuantización menor (q4_k_m)
- Reduce `num_ctx` si es muy alto
- Verifica que Ollama esté usando GPU

## Resumen del flujo completo

```bash
# 1. Dataset listo en data/
make train          # Entrenar

# 2. Fusionar adaptador
make merge          # Merge LoRA

# 3. Convertir a GGUF
make convert        # Crear GGUF

# 4. Desplegar
make ollama-up      # Iniciar Ollama
make deploy         # Crear modelo

# 5. Usar
make chat           # Probar
```

¡Felicidades! Ahora tienes tu propio modelo fine-tuned funcionando localmente con Ollama.
