.PHONY: build train merge convert deploy chat jupyter shell clean help

# Variables
DATASET ?= sample_dataset.csv
MODEL_NAME ?= mi-modelo-finetuned
BASE_MODEL ?= unsloth/Llama-3.2-1B-Instruct
QUANTIZATION ?= q4_k_m

help:
	@echo "Comandos disponibles:"
	@echo "  make build      - Construir imágenes Docker"
	@echo "  make train      - Ejecutar entrenamiento (DATASET=archivo.csv)"
	@echo "  make merge      - Fusionar LoRA con modelo base"
	@echo "  make convert    - Convertir a GGUF (QUANTIZATION=q4_k_m)"
	@echo "  make deploy     - Crear modelo en Ollama (MODEL_NAME=nombre)"
	@echo "  make chat       - Probar el modelo en Ollama"
	@echo "  make jupyter    - Iniciar Jupyter Notebook"
	@echo "  make shell      - Shell en contenedor de training"
	@echo "  make ollama-up  - Iniciar servidor Ollama"
	@echo "  make clean      - Limpiar artefactos"

build:
	docker compose build

train:
	docker compose run --rm training python /app/scripts/train.py \
		--dataset /app/data/$(DATASET) \
		--base-model $(BASE_MODEL) \
		--output-dir /app/models/lora_adapter

merge:
	docker compose run --rm training python /app/scripts/merge_lora.py \
		--base-model $(BASE_MODEL) \
		--lora-path /app/models/lora_adapter \
		--output-dir /app/models/merged

convert:
	docker compose run --rm training python /app/scripts/convert_to_gguf.py \
		--model-path /app/models/merged \
		--output-path /app/models/gguf/$(MODEL_NAME).gguf \
		--quantization $(QUANTIZATION)

deploy:
	@echo "FROM /models/$(MODEL_NAME).gguf" > ollama/Modelfile
	@cat ollama/Modelfile.template >> ollama/Modelfile
	docker compose exec ollama ollama create $(MODEL_NAME) -f /ollama/Modelfile
	@echo "Modelo $(MODEL_NAME) creado en Ollama"

chat:
	docker compose exec ollama ollama run $(MODEL_NAME)

jupyter:
	docker compose --profile jupyter up jupyter

shell:
	docker compose run --rm training bash

ollama-up:
	docker compose up -d ollama

ollama-down:
	docker compose down ollama

clean:
	rm -rf models/lora_adapter/*
	rm -rf models/merged/*
	rm -rf models/gguf/*
	rm -f ollama/Modelfile
	@echo "Artefactos limpiados"

# Crear directorios necesarios
init:
	mkdir -p models/lora_adapter models/merged models/gguf models/huggingface
	mkdir -p data notebooks
	@echo "Directorios creados"
