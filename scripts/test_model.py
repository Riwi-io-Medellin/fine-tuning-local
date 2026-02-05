#!/usr/bin/env python3
"""
Script para probar el modelo entrenado
Compara respuestas antes y después del fine-tuning
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Probar modelo entrenado')
    parser.add_argument('--model-path', type=str, 
                        default='/app/models/lora_adapter',
                        help='Ruta al modelo/adaptador')
    parser.add_argument('--base-model', type=str,
                        default='unsloth/Llama-3.2-1B-Instruct',
                        help='Modelo base para comparación')
    parser.add_argument('--prompt', type=str,
                        default='¿Qué es fine-tuning?',
                        help='Prompt de prueba')
    return parser.parse_args()


def generate_response(model, tokenizer, prompt, max_tokens=256):
    """Genera una respuesta del modelo"""
    
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        input_text = f"### Instrucción:\n{prompt}\n\n### Respuesta:\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta
    if "### Respuesta:" in response:
        response = response.split("### Respuesta:")[-1].strip()
    elif "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    elif "assistant" in response.lower():
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


def main():
    args = parse_args()
    
    print("="*60)
    print("Prueba de Modelo Fine-tuned")
    print("="*60)
    
    try:
        from unsloth import FastLanguageModel
        
        # Cargar modelo fine-tuned
        print(f"\nCargando modelo: {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"\nCargando modelo base: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        print(f"Cargando adaptador: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    
    print(f"\nPrompt: {args.prompt}")
    print("-"*40)
    
    response = generate_response(model, tokenizer, args.prompt)
    print(f"Respuesta:\n{response}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
