import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with a base model.")
    parser.add_argument("--base_model_id", type=str, required=True, help="Hugging Face model ID of the base model (e.g., 'mistralai/Mistral-7B-v0.1').")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter checkpoint directory (e.g., 'mistral-qlora-output/checkpoint-xxx').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model and tokenizer.")
    parser.add_argument("--torch_dtype", type=str, default="torch.float16", help="Torch dtype for loading the base model (e.g., 'torch.float16', 'torch.bfloat16', 'torch.float32').")

    args = parser.parse_args()

    # Resolve torch_dtype string to actual torch.dtype
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    resolved_torch_dtype = dtype_map.get(args.torch_dtype)
    if resolved_torch_dtype is None:
        print(f"Warning: Invalid torch_dtype '{args.torch_dtype}'. Defaulting to torch.float16.")
        resolved_torch_dtype = torch.float16

    print(f"Loading base model: {args.base_model_id} with dtype: {resolved_torch_dtype}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=resolved_torch_dtype,
        device_map="auto", # Use "auto" or "cpu" if GPU memory is an issue for merging
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {args.adapter_path}")
    # Ensure the adapter is loaded onto the same device as the base model,
    # device_map="auto" for PeftModel should handle this.
    model_to_merge = PeftModel.from_pretrained(base_model, args.adapter_path, device_map="auto")

    print("Merging LoRA adapters into the base model...")
    merged_model = model_to_merge.merge_and_unload()
    print("Merging complete.")

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)

    print(f"Loading tokenizer for base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)

    print(f"Saving tokenizer to: {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)

    print(f"Successfully merged model and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
