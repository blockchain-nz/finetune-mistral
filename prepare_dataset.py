import json
from datasets import Dataset, DatasetDict
import argparse
import os

def load_qa_from_json(json_path: str) -> list:
    """Loads Q&A pairs from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Prepare Q&A dataset for Hugging Face training.")
    parser.add_argument("--input_json", type=str, default="output/qa_dataset.json", help="Path to the input Q&A JSON file.")
    parser.add_argument("--output_dir", type=str, default="output/mistral_qa_dataset", help="Directory to save the processed dataset.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to use for the test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")

    args = parser.parse_args()

    print(f"Loading Q&A pairs from: {args.input_json}")
    try:
        qa_pairs = load_qa_from_json(args.input_json)
    except Exception as e:
        print(f"Error loading Q&A JSON: {e}")
        return

    if not qa_pairs:
        print("No Q&A pairs found. Exiting.")
        return

    print(f"Loaded {len(qa_pairs)} Q&A pairs.")

    # Convert list of dicts to Hugging Face Dataset
    try:
        dataset = Dataset.from_list(qa_pairs)
    except Exception as e:
        print(f"Error creating Dataset from list: {e}")
        return

    print("Successfully created Hugging Face Dataset.")

    # Split dataset into train and test
    if args.test_size > 0 and args.test_size < 1:
        print(f"Splitting dataset into train and test sets (test_size={args.test_size}, seed={args.seed}).")
        dataset_dict = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        print(f"Train set size: {len(dataset_dict['train'])}")
        print(f"Test set size: {len(dataset_dict['test'])}")
    elif args.test_size == 0:
        print("No test split requested (test_size=0). Using entire dataset as train set.")
        dataset_dict = DatasetDict({'train': dataset})
        print(f"Train set size: {len(dataset_dict['train'])}")
    else:
        print(f"Invalid test_size: {args.test_size}. Must be between 0.0 and 1.0. Using entire dataset as train set.")
        dataset_dict = DatasetDict({'train': dataset})
        print(f"Train set size: {len(dataset_dict['train'])}")

    def clear_input(example):
        example["input"] = ""
        return example

    dataset_dict = DatasetDict({
        split: dataset_dict[split].map(clear_input, batched=False)
        for split in dataset_dict
    })

    # Save dataset to disk
    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        dataset_dict.save_to_disk(args.output_dir)
        print(f"Successfully saved dataset to: {args.output_dir}")
    except Exception as e:
        print(f"Error saving dataset to disk: {e}")

if __name__ == "__main__":
    main()
