from datasets import load_dataset
from datasets import Image
import pandas as pd


# Load the jsonl file and create a copy with only the first two columns
df = pd.read_json("train2.jsonl", lines=True)
df = df[["image", "markdown"]]

# Convert to jsonl
df.to_json("train3.jsonl", orient="records", lines=True, force_ascii=False)
dataset = load_dataset("json", data_files="train3.jsonl", split="train")
dataset = dataset.cast_column("image", Image())

# Push the dataset to the hub
dataset.push_to_hub("powow/swe-data")
