from datasets import load_from_disk
ds = load_from_disk("output/mistral_qa_dataset")
print(ds["train"][:5])
