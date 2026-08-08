from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json


MODEL = "aubmindlab/bert-base-arabertv02"


# Load data

dataset = load_dataset(
    "json",
    data_files={
        "train": "dataset/intent/train.json",
        "validation": "dataset/intent/validation.json"
    }
)


# Encode labels

labels = set(
    dataset["train"]["label"]
)

label_encoder = LabelEncoder()

label_encoder.fit(
    list(labels)
)


def encode(example):

    example["label"] = (
        label_encoder
        .transform(
            [example["label"]]
        )[0]
    )

    return example


dataset = dataset.map(
    encode
)


# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL
)


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


dataset = dataset.map(
    tokenize,
    batched=True
)


dataset = dataset.remove_columns(
    ["text"]
)


dataset.set_format(
    "torch"
)


# Model

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(labels)
)


# Training

args = TrainingArguments(
    output_dir="./models/intent",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    fp16=True,
    logging_steps=10
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)


trainer.train()


model.save_pretrained(
    "./models/intent"
)

tokenizer.save_pretrained(
    "./models/intent"
)


with open(
    "./models/intent/labels.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        list(label_encoder.classes_),
        f,
        ensure_ascii=False
    )


print("Done")