# ¿Qué es Fine-tuning?

## La idea central

Imagina que tienes un empleado nuevo muy inteligente: sabe de todo un poco, pero no conoce los procesos específicos de tu empresa. **Fine-tuning es entrenar a ese empleado** con tus documentos, tu estilo, tus casos reales.

Un LLM pre-entrenado (como Llama, Mistral, GPT) ha sido entrenado con billones de tokens de internet. Sabe mucho, pero de forma general. Fine-tuning lo **especializa**.

## Comparación de técnicas

| Técnica | Qué hace | Cuándo usarla |
|---------|----------|---------------|
| **Prompting** | Das instrucciones en cada request | Tareas simples, prototipos rápidos |
| **RAG** | Buscas información y la pasas al modelo | Conocimiento que cambia, bases de datos |
| **Fine-tuning** | Modificas los pesos del modelo | Estilo consistente, tareas especializadas |

## Cuándo usar Fine-tuning

**Sí usa fine-tuning cuando:**
- Necesitas un estilo o formato muy específico
- Tienes datos propietarios de alta calidad
- Quieres respuestas consistentes sin prompts largos
- El dominio es técnico y especializado

**No uses fine-tuning cuando:**
- Solo necesitas acceso a información (usa RAG)
- Los datos cambian frecuentemente
- No tienes suficientes ejemplos de calidad (<100)
- Un buen prompt resuelve el problema

## El proceso en resumen

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Modelo Base │ ──▶ │ Entrenamiento│ ──▶ │ Modelo      │
│ (Llama 3)   │     │ con tus datos│     │ Especializado│
└─────────────┘     └──────────────┘     └─────────────┘
```

1. **Modelo base**: Un LLM pre-entrenado con conocimiento general
2. **Tus datos**: Ejemplos de cómo quieres que responda
3. **Resultado**: Un modelo que combina el conocimiento general con tu especialización

## Lo que aprenderás en este tutorial

1. **LoRA/QLoRA**: Técnicas para fine-tunear sin explotar tu GPU
2. **Preparación de datos**: Cómo estructurar tus ejemplos
3. **Entrenamiento**: Ejecutar el proceso paso a paso
4. **Conversión**: Transformar el modelo a formato eficiente
5. **Despliegue**: Usar tu modelo con Ollama

## Mitos comunes

### "Fine-tuning agrega conocimiento nuevo"

**Parcialmente falso**. Fine-tuning es mejor para:
- Cambiar el estilo de respuesta
- Seguir formatos específicos
- Reforzar patrones existentes

Para conocimiento nuevo, RAG suele funcionar mejor.

### "Necesito miles de ejemplos"

**Depende**. Para estilo y formato, 200-500 ejemplos de alta calidad pueden ser suficientes. La calidad > cantidad.

### "Necesito una GPU enorme"

**Ya no**. Con QLoRA puedes fine-tunear modelos de 7B en una RTX 3060 de 12GB. Este tutorial está diseñado para hardware accesible.

## Siguiente paso

Continúa con [02 - LoRA y QLoRA](02-lora-qlora-explicado.md) para entender cómo funcionan estas técnicas.
