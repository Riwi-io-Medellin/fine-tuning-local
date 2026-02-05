#!/bin/bash
# Script para crear modelo en Ollama a partir de GGUF

set -e

MODEL_NAME=${1:-"mi-modelo-finetuned"}
GGUF_PATH=${2:-"/models/${MODEL_NAME}.gguf"}

echo "Creando modelo: $MODEL_NAME"
echo "Desde GGUF: $GGUF_PATH"

# Crear Modelfile temporal
cat > /tmp/Modelfile << EOF
FROM $GGUF_PATH

$(cat /ollama/Modelfile.template)
EOF

# Crear modelo en Ollama
ollama create "$MODEL_NAME" -f /tmp/Modelfile

echo "Modelo $MODEL_NAME creado exitosamente"
echo "Ejecuta: ollama run $MODEL_NAME"
