from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import json
import shutil
import os
from collections import Counter

print("Starting training script...")

# Remove old model and logs folders if they exist
if os.path.exists("./model"):
    print("Removing old model directory...")
    shutil.rmtree("./model")

if os.path.exists("./logs"):
    print("Removing old logs directory...")
    shutil.rmtree("./logs")

# Load data
with open("data.json") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} samples")

# Prepare label mappings
label_texts = list(set(item["output"] for item in raw_data))
label2id = {text: i for i, text in enumerate(label_texts)}
id2label = {i: text for text, i in label2id.items()}

data = {
    "text": [item["input"] for item in raw_data],
    "labels": [label2id[item["output"]] for item in raw_data]
}

# Create dataset and shuffle
dataset = Dataset.from_dict(data).shuffle(seed=42)

print("Label distribution:", Counter(data["labels"]))
print(f"Dataset created with {len(dataset)} examples")

# Load tokenizer and define preprocess function
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = example["labels"]
    return encoding

tokenized_dataset = dataset.map(preprocess, batched=True)

# Load model with correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments without unsupported parameters
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model and tokenizer...")
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

print("Training complete!")
