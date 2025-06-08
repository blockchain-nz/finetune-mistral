import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    adapter_path = "mistral-qlora-output/final_model"

    # 1) QLoRA 4-bit config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2) Load the 4-bit base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) Attach your fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,      # <- the base model object
        adapter_path,    # <- your adapter folder
        torch_dtype=torch.float16,
    )
    model.eval()

    # 4) Load tokenizer (from the same adapter folder, where you saved it)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 5) Build prompt exactly as you trained
    instruction = "What is the Kyle Yu email address?"
    prompt = (
        "### Instruction:\n"
        f"{instruction}\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 6) Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 7) Strip off prompt tokens and decode
    gen = out[0][ inputs.input_ids.shape[1] : ]
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    print(answer)

if __name__ == "__main__":
    main()
