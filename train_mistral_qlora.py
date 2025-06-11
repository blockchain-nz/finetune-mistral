import argparse
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from email_sender import send_notification
from transformers import default_data_collator


def preprocess_function(examples, tokenizer, max_length=1024):
    input_ids, labels = [], []

    for instr, out in zip(examples["instruction"], examples["output"]):
        # build the prompt exactly as you use it at inference:
        prompt = f"### Instruction:\n{instr}\n### Response:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(out, add_special_tokens=False)["input_ids"]

        # concat prompt + output; then pad/truncate:
        ids = prompt_ids + output_ids + [tokenizer.eos_token_id]
        if len(ids) > max_length:
            ids = ids[:max_length]

        # create labels: mask prompt_ids, keep only output_ids+eos for loss
        lbls = [-100] * len(prompt_ids) + output_ids + [tokenizer.eos_token_id]
        lbls = lbls[: len(ids)]  # align lengths

        # pad up to max_length
        padding_length = max_length - len(ids)
        ids = ids + [tokenizer.pad_token_id] * padding_length
        lbls = lbls + [-100] * padding_length

        input_ids.append(ids)
        labels.append(lbls)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([[1]*len(ids) + [0]* (max_length-len(ids))
                                        for ids in input_ids], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch



def main():
    load_dotenv()  # Load .env file
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face using HF_TOKEN...")
        login(hf_token)
    else:
        print("Warning: HF_TOKEN not set. You may not be able to access gated models.")

    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B with QLoRA")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/mistral-qlora-output", help="Directory to save the model and checkpoints")

    parser.add_argument("--load_in_4bit", action='store_true', default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--fp16", action='store_true', default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    if args.load_in_4bit:
        print("Preparing model for QLoRA...")
        model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from disk: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    print("Tokenization done.")
    print(f"Train set size: {len(tokenized_dataset['train'])}")
    if 'test' in tokenized_dataset:
        print(f"Test set size: {len(tokenized_dataset['test'])}")

    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_total_limit=2,
        load_best_model_at_end=args.evaluation_strategy != "no",
        report_to="tensorboard",
        seed=args.seed,
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("test"),
        tokenizer=tokenizer,
        # data_collator=data_collator
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    final_save_path = f"{args.output_dir}/final_model"
    print(f"Saving model and tokenizer to {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Model and tokenizer saved.")
    send_notification("Model Training Completed", "<p>Training has finished successfully.</p>")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)


if __name__ == "__main__":
    main()
