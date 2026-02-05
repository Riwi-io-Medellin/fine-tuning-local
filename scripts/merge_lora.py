#!/usr/bin/env python3
"""
Script para fusionar adaptadores LoRA con el modelo base
Produce un modelo completo listo para conversión a GGUF
"""

import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fusionar LoRA con modelo base')
    parser.add_argument('--base-model', type=str, 
                        default='unsloth/Llama-3.2-1B-Instruct',
                        help='Modelo base original')
    parser.add_argument('--lora-path', type=str, 
                        default='/app/models/lora_adapter',
                        help='Ruta al adaptador LoRA entrenado')
    parser.add_argument('--output-dir', type=str, 
                        default='/app/models/merged',
                        help='Directorio para el modelo fusionado')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Fusión de LoRA con Modelo Base")
    print("="*60)
    
    try:
        # Intentar con Unsloth primero
        from unsloth import FastLanguageModel
        
        print(f"\nCargando modelo base: {args.base_model}")
        print(f"Cargando adaptador LoRA: {args.lora_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.lora_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Cargar en FP16 para merge
        )
        
        print("\nFusionando adaptador con modelo base...")
        
        # Guardar modelo fusionado
        print(f"Guardando en: {args.output_dir}")
        model.save_pretrained_merged(
            args.output_dir, 
            tokenizer,
            save_method="merged_16bit",
        )
        
    except ImportError:
        # Fallback a PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print("Usando PEFT para fusión...")
        
        print(f"\nCargando modelo base: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        print(f"Cargando adaptador LoRA: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        
        print("\nFusionando adaptador con modelo base...")
        model = model.merge_and_unload()
        
        print(f"Guardando en: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "="*60)
    print("Fusión completada!")
    print("="*60)
    print(f"\nModelo fusionado guardado en: {args.output_dir}")
    print("Siguiente paso: ejecuta 'make convert' para convertir a GGUF")


if __name__ == "__main__":
    main()
