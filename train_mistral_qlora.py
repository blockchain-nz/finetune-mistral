import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def preprocess_function(examples, tokenizer, max_length=1024):
    """
    Prepares the input by formatting instructions and responses,
    then tokenizing them.
    """
    prompts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        # Input field is not used in this specific prompt format from the user's example
        # If 'input' is relevant, the prompt format should be adjusted.
        # prompt = f"### Instruction:\n{instruction}\n### Input:\n{examples['input']}\n### Response:\n{output}"
        prompt = f"### Instruction:\n{instruction}\n### Response:\n{output}"
        prompts.append(prompt)

    # Tokenize the prompts
    tokenized_inputs = tokenizer(
        prompts,
        truncation=True,
        padding="max_length", # Pad to max_length
        max_length=max_length,
        return_attention_mask=True # Return attention mask
    )

    # For causal LM, labels are typically the input_ids themselves.
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B with QLoRA")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1", help="Hugging Face model ID for Mistral 7B.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preprocessed Hugging Face dataset directory.")
    parser.add_argument("--output_dir", type=str, default="mistral-qlora-output", help="Directory to save training outputs (checkpoints, logs).")

    # QLoRA specific arguments
    parser.add_argument("--load_in_4bit", action='store_true', default=True, help="Load model in 4-bit for QLoRA.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Modules to apply LoRA to.")

    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--fp16", action='store_true', default=True, help="Whether to use 16-bit (mixed) precision training.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy to adopt during training (e.g., 'no', 'epoch', 'steps').")
    parser.add_argument("--evaluation_strategy", type=str, default="no", help="Evaluation strategy to adopt during training (e.g., 'no', 'epoch', 'steps').")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for tokenization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    print(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token to EOS token

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",  # Automatically distributes model across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.fp16 else torch.float32 # Match dtype with fp16 training
    )

    if args.load_in_4bit:
        print("Preparing model for k-bit training (QLoRA).")
        model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA.")
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

    # The user's original script mapped over the dataset directly.
    # We will create a partial function for the map.
    # The original preprocess function:
    # def preprocess(example):
    #     prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    #     input_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
    #     input_ids["labels"] = input_ids["input_ids"].copy()
    #     return input_ids
    # tokenized = dataset.map(preprocess, batched=False)
    #
    # We will use the preprocess_function defined above, which expects batched input by default from .map()
    # but we can make it work like the original by setting batched=True and handling lists of examples.
    # Or, ensure the function is called per example if batched=False.
    # Hugging Face `map` is more efficient with `batched=True`.

    print("Tokenizing dataset...")
    # The preprocess_function is designed to work with map's default batched=True
    # It iterates through examples['instruction'] and examples['output'] which are lists in batched mode.
    tokenized_dataset = dataset.map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        batched=True, # Process multiple examples at once
        remove_columns=dataset["train"].column_names # Remove original columns to avoid issues with collator
    )
    print("Dataset tokenized.")
    print(f"Tokenized train dataset size: {len(tokenized_dataset['train'])}")
    if 'test' in tokenized_dataset:
        print(f"Tokenized test dataset size: {len(tokenized_dataset['test'])}")


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Setting up TrainingArguments.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{args.output_dir}/logs", # Specify logging directory
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        # eval_steps is needed if evaluation_strategy is 'steps'
        # save_steps is needed if save_strategy is 'steps'
        save_total_limit=2, # Optional: limits the total number of checkpoints
        load_best_model_at_end=True if args.evaluation_strategy != "no" else False,
        report_to="tensorboard", # Optional: report metrics to tensorboard
        seed=args.seed,
    )

    print("Initializing Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("test"), # Use .get() in case 'test' split is not present
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print("Training finished.")

    final_save_path = f"{args.output_dir}/final_model"
    print(f"Saving final model and tokenizer to {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
