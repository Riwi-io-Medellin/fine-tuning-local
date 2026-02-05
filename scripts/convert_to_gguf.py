#!/usr/bin/env python3
"""
Script para convertir modelos HuggingFace a formato GGUF
GGUF es el formato usado por llama.cpp y Ollama
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Convertir modelo a GGUF')
    parser.add_argument('--model-path', type=str, 
                        default='/app/models/merged',
                        help='Ruta al modelo fusionado')
    parser.add_argument('--output-path', type=str, 
                        default='/app/models/gguf/model.gguf',
                        help='Ruta de salida para el archivo GGUF')
    parser.add_argument('--quantization', type=str, 
                        default='q4_k_m',
                        choices=['f16', 'f32', 'q4_0', 'q4_1', 'q4_k_m', 
                                'q4_k_s', 'q5_0', 'q5_1', 'q5_k_m', 'q5_k_s',
                                'q6_k', 'q8_0'],
                        help='Tipo de cuantización')
    return parser.parse_args()


def find_llama_cpp():
    """Encuentra el directorio de llama.cpp"""
    possible_paths = [
        os.environ.get('LLAMA_CPP_DIR', ''),
        '/opt/llama.cpp',
        os.path.expanduser('~/llama.cpp'),
        './llama.cpp',
    ]
    
    for path in possible_paths:
        if path and os.path.exists(os.path.join(path, 'convert_hf_to_gguf.py')):
            return path
    
    return None


def convert_to_gguf(model_path, output_path, quantization):
    """Convierte el modelo a GGUF usando llama.cpp"""
    
    llama_cpp_dir = find_llama_cpp()
    
    if llama_cpp_dir:
        # Usar script de llama.cpp
        convert_script = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')
        
        print(f"Usando llama.cpp en: {llama_cpp_dir}")
        
        # Crear directorio de salida
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Primero convertir a GGUF F16
        temp_output = output_path.replace('.gguf', '_f16.gguf')
        
        cmd = [
            sys.executable,
            convert_script,
            model_path,
            '--outfile', temp_output,
            '--outtype', 'f16',
        ]
        
        print(f"Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error en conversión: {result.stderr}")
            return False
        
        print(f"Modelo convertido a F16: {temp_output}")
        
        # Cuantizar si es necesario
        if quantization != 'f16':
            quantize_bin = os.path.join(llama_cpp_dir, 'llama-quantize')
            
            if not os.path.exists(quantize_bin):
                # Intentar compilar
                print("Compilando llama-quantize...")
                subprocess.run(['make', 'llama-quantize'], cwd=llama_cpp_dir)
            
            if os.path.exists(quantize_bin):
                cmd = [quantize_bin, temp_output, output_path, quantization.upper()]
                print(f"Cuantizando a {quantization}...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    os.remove(temp_output)
                    print(f"Modelo cuantizado: {output_path}")
                else:
                    print(f"Error en cuantización: {result.stderr}")
                    # Usar el F16 como fallback
                    os.rename(temp_output, output_path)
            else:
                print("llama-quantize no disponible, usando F16")
                os.rename(temp_output, output_path)
        else:
            os.rename(temp_output, output_path)
        
        return True
    
    else:
        # Intentar con Unsloth
        try:
            from unsloth import FastLanguageModel
            
            print("Usando Unsloth para conversión GGUF...")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=False,
            )
            
            # Unsloth tiene método directo para GGUF
            quantization_map = {
                'q4_k_m': 'q4_k_m',
                'q8_0': 'q8_0',
                'f16': 'f16',
            }
            
            quant = quantization_map.get(quantization, 'q4_k_m')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            model.save_pretrained_gguf(
                os.path.dirname(output_path),
                tokenizer,
                quantization_method=quant,
            )
            
            # Renombrar al nombre deseado
            default_name = os.path.join(os.path.dirname(output_path), 
                                       f"unsloth.{quant.upper()}.gguf")
            if os.path.exists(default_name):
                os.rename(default_name, output_path)
            
            print(f"Modelo guardado: {output_path}")
            return True
            
        except ImportError:
            print("ERROR: No se encontró llama.cpp ni Unsloth")
            print("Instala llama.cpp o Unsloth para conversión GGUF")
            return False


def main():
    args = parse_args()
    
    print("="*60)
    print("Conversión a GGUF")
    print("="*60)
    
    print(f"\nModelo fuente: {args.model_path}")
    print(f"Archivo destino: {args.output_path}")
    print(f"Cuantización: {args.quantization}")
    
    # Verificar que existe el modelo
    if not os.path.exists(args.model_path):
        print(f"\nERROR: No se encontró el modelo en {args.model_path}")
        print("Ejecuta primero 'make merge' para crear el modelo fusionado")
        sys.exit(1)
    
    success = convert_to_gguf(
        args.model_path, 
        args.output_path, 
        args.quantization
    )
    
    if success:
        print("\n" + "="*60)
        print("Conversión completada!")
        print("="*60)
        print(f"\nArchivo GGUF: {args.output_path}")
        print("Siguiente paso: ejecuta 'make deploy' para crear el modelo en Ollama")
    else:
        print("\nError en la conversión")
        sys.exit(1)


if __name__ == "__main__":
    main()
