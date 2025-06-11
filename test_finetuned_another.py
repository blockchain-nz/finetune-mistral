from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# load base + adapter
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                          bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                            quantization_config=quant,
                                            device_map="auto",
                                            trust_remote_code=True)
model = PeftModel.from_pretrained(base, "output/mistral-qlora-output/final_model",
                                  torch_dtype=torch.float16, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("output/mistral-qlora-output/final_model",
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

prompt = "### Instruction:\nWhat is the candidate's email address?\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                     eos_token_id=tokenizer.eos_token_id)
answer = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)  # should be superkyle1112@gmail.com
