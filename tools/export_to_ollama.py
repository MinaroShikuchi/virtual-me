#!/usr/bin/env python3
"""
tools/export_to_ollama.py

Loads a trained LoRA adapter and exports it as a GGUF model 
that can be directly imported into Ollama.
"""

import sys
import subprocess

def export_model(adapter_path="./models/my-lora", out_path="./models/my-lora-gguf", quant_method="q4_k_m", ollama_name=None):
    from unsloth import FastLanguageModel
    print(f"Loading adapter from {adapter_path}...")
    
    # 1. Load the fine-tuned adapter 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path, 
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"\nExporting to GGUF format ({quant_method})...")
    print("This will take a few minutes as it merges the LoRA and quantizes the model.")
    
    # 2. Save to GGUF format (this internally uses llama.cpp)
    # Unsloth will create out_path + "_gguf" directory automatically
    model.save_pretrained_gguf(out_path, tokenizer, quantization_method = quant_method)
    
    gguf_dir = f"{out_path}_gguf"
    print(f"\n✅ Export complete! Your GGUF files and Modelfile are in: {gguf_dir}")

    # 3. Import into Ollama directly if requested
    if ollama_name:
        print(f"\nRegistering model '{ollama_name}' in Ollama...")
        try:
            subprocess.run(["ollama", "create", ollama_name, "-f", "Modelfile"], cwd=gguf_dir, check=True)
            print(f"✅ Successfully compiled '{ollama_name}' into Ollama!")
            print(f"To run it: ollama run {ollama_name}")
        except Exception as e:
            print(f"❌ Failed to run 'ollama create': {e}")
            print(f"You can manually run 'ollama create {ollama_name} -f Modelfile' from {gguf_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export LoRA to Ollama GGUF")
    parser.add_argument("--adapter", default="./models/my-lora", help="Path to your trained LoRA adapter")
    parser.add_argument("--output", default="./models/my-lora-gguf", help="Output directory for GGUF file")
    parser.add_argument("--quant", default="q4_k_m", help="Quantization method (e.g. q4_k_m, q8_0, f16)")
    parser.add_argument("--ollama-name", default=None, help="If provided, automatically run 'ollama create' and bind this name.")
    args = parser.parse_args()
    
    export_model(args.adapter, args.output, args.quant, args.ollama_name)
