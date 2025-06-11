from peft import PeftConfig
config = PeftConfig.from_pretrained("output/mistral-qlora-output/final_model")
print(config)
