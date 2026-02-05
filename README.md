# Fine-Tuning de LLMs en Local con Ollama

Tutorial práctico y educativo para aprender fine-tuning de modelos de lenguaje usando LoRA/QLoRA, con despliegue en Ollama. **100% dockerizado**.

## ¿Qué vas a aprender?

- La diferencia entre prompting, RAG y fine-tuning
- Cómo funcionan LoRA y QLoRA
- Entrenar un LLM en tu máquina local
- Preparar datasets para entrenamiento
- Convertir modelos a formato GGUF
- Desplegar modelos personalizados con Ollama

## Requisitos

### Hardware

| Configuración | Resultado |
|--------------|-----------|
| RTX 3060 12GB | Modelos hasta 7B |
| RTX 3090/4090 | Excelente para 7B-13B |
| Sin GPU (CPU) | Funcional pero lento |
| 16GB+ RAM | Recomendado |

### Software

- Docker + Docker Compose
- NVIDIA Container Toolkit (si tienes GPU)
- Git

## Quick Start

```bash
# 1. Clonar el repo
git clone https://github.com/tu-usuario/fine-tuning-local.git
cd fine-tuning-local

# 2. Construir imágenes
make build

# 3. Entrenar con dataset de ejemplo
make train

# 4. Convertir a GGUF
make convert

# 5. Desplegar en Ollama
make deploy

# 6. Probar el modelo
make chat
```

## Estructura del Proyecto

```
.
├── docker-compose.yml      # Servicios Docker
├── Makefile               # Comandos simplificados
├── docs/                  # Documentación teórica
│   ├── 01-que-es-fine-tuning.md
│   ├── 02-lora-qlora-explicado.md
│   ├── 03-preparacion-datos.md
│   ├── 04-entrenamiento-paso-a-paso.md
│   ├── 05-conversion-gguf.md
│   └── 06-despliegue-ollama.md
├── scripts/               # Scripts de entrenamiento
│   ├── train.py
│   ├── merge_lora.py
│   └── convert_to_gguf.py
├── data/                  # Datasets
│   └── sample_dataset.csv
├── models/                # Modelos entrenados
└── ollama/                # Configuración Ollama
    └── Modelfile.template
```

## Documentación

| Doc | Descripción |
|-----|-------------|
| [01 - ¿Qué es Fine-tuning?](docs/01-que-es-fine-tuning.md) | Fundamentos y cuándo usarlo |
| [02 - LoRA y QLoRA](docs/02-lora-qlora-explicado.md) | Cómo funcionan estas técnicas |
| [03 - Preparación de Datos](docs/03-preparacion-datos.md) | Formato y calidad del dataset |
| [04 - Entrenamiento](docs/04-entrenamiento-paso-a-paso.md) | Proceso paso a paso |
| [05 - Conversión GGUF](docs/05-conversion-gguf.md) | Formato para Ollama |
| [06 - Despliegue Ollama](docs/06-despliegue-ollama.md) | Servir tu modelo |

## Comandos Disponibles

```bash
make build      # Construir imágenes Docker
make train      # Ejecutar entrenamiento
make merge      # Fusionar LoRA con modelo base
make convert    # Convertir a GGUF
make deploy     # Crear modelo en Ollama
make chat       # Probar el modelo
make jupyter    # Iniciar Jupyter Notebook
make shell      # Shell en contenedor de training
make clean      # Limpiar artefactos
make help       # Ver todos los comandos
```

## Usar tu propio dataset

1. Crea un CSV con columnas `instruction` y `response`:

```csv
instruction,response
"¿Qué es un callback en Rails?","Un callback en Rails es un método..."
"Explica ActiveRecord","ActiveRecord es el ORM de Rails..."
```

2. Guárdalo en `data/mi_dataset.csv`

3. Ejecuta:

```bash
make train DATASET=mi_dataset.csv
```

## Habilitar GPU (opcional)

Por defecto el proyecto usa **CPU**. Si tienes GPU NVIDIA, edita `docker-compose.yml` y descomenta las secciones marcadas con `CONFIGURACIÓN GPU`:

```yaml
# ============================================================
# CONFIGURACIÓN GPU (descomentar si tienes NVIDIA GPU)
# ============================================================
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
# ============================================================
```

También necesitas:
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado
- Drivers NVIDIA actualizados

El entrenamiento en CPU es más lento pero funciona para datasets pequeños (~500 ejemplos).

## Errores Comunes

| Error | Solución |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` o usa modelo más pequeño |
| Model not found | Verifica conexión a internet para descargar modelo base |
| Permission denied | Ejecuta con `sudo` o ajusta permisos de Docker |
| Port 11434 already in use | Ollama ya está corriendo en el sistema. Ver solución abajo |

### Error: Puerto 11434 en uso

Si al ejecutar `docker compose up` ves este error:

```
Error response from daemon: failed to bind host port for 0.0.0.0:11434
address already in use
```

**Causa:** Ollama está corriendo como servicio del sistema y ocupa el puerto 11434.

**Solución:**

```bash
# Detener el servicio de Ollama
sudo systemctl stop ollama

# (Opcional) Deshabilitar el inicio automático
sudo systemctl disable ollama

# Ahora puedes ejecutar docker compose
docker compose up
```

**Alternativa:** Si prefieres usar el Ollama del sistema en lugar del contenedor, modifica `docker-compose.yml` cambiando la variable `OLLAMA_HOST` del servicio trainer a `http://host.docker.internal:11434`.

## Objetivo del Proyecto

Este repo **no busca** entrenar el mejor modelo del mundo.

**Busca** que:
- Entiendas cómo funciona el fine-tuning
- Lo ejecutes en tu máquina
- Experimentes con tus propios datos
- Aprendas haciendo, no solo leyendo

## Licencia

MIT
