#!/usr/bin/env python3
"""
Script de entrenamiento de LLMs con LoRA/QLoRA
Fine-tuning de modelos usando HuggingFace TRL
"""

import argparse
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning de LLM con LoRA')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Ruta al archivo CSV del dataset')
    parser.add_argument('--base-model', type=str, 
                        default='unsloth/Llama-3.2-1B-Instruct',
                        help='Modelo base a usar')
    parser.add_argument('--output-dir', type=str, 
                        default='/app/models/lora_adapter',
                        help='Directorio para guardar el adaptador LoRA')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='Longitud máxima de secuencia')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Número de epochs de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size por dispositivo')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    return parser.parse_args()


def check_hardware():
    """Detecta el hardware disponible"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
        return "cuda"
    else:
        print("No se detectó GPU. Usando CPU (será más lento)")
        return "cpu"


def load_model_and_tokenizer(model_name, device):
    """Carga el modelo base con LoRA"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print("Cargando modelo con PEFT...")
    
    # Configuración de cuantización 4-bit
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Preparar para entrenamiento si hay cuantización
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def format_instruction(row, tokenizer):
    """Formatea el dataset al estilo chat"""
    instruction = row.get('instruction', row.get('Context', ''))
    response = row.get('response', row.get('Response', ''))
    
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n{response}"
    
    return {"text": text}


def main():
    args = parse_args()
    
    print("="*60)
    print("Fine-Tuning de LLM con LoRA")
    print("="*60)
    
    device = check_hardware()
    
    print(f"\nCargando modelo: {args.base_model}")
    model, tokenizer = load_model_and_tokenizer(args.base_model, device)
    
    # Mostrar parámetros entrenables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Cargar dataset
    print(f"\nCargando dataset: {args.dataset}")
    dataset = load_dataset('csv', data_files=args.dataset)
    
    dataset = dataset.map(
        lambda row: format_instruction(row, tokenizer),
        remove_columns=dataset['train'].column_names
    )
    
    print(f"Ejemplos: {len(dataset['train'])}")
    
    # Configurar entrenamiento con SFTConfig
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=50,
        dataset_text_field="text",
        packing=False,
        fp16=False,
        bf16=False,
        report_to="none",
    )
    
    # Crear trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        processing_class=tokenizer,
        args=sft_config,
    )
    
    # Entrenar
    print("\n" + "="*60)
    print("Iniciando entrenamiento...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Guardar
    print(f"\nGuardando adaptador LoRA en: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "="*60)
    print("Entrenamiento completado!")
    print("="*60)
    print(f"\nSiguiente paso: ejecuta 'make merge'")


if __name__ == "__main__":
    main()
