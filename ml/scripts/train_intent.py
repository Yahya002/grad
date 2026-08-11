import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ML_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

TRAIN_FILE = ML_DIR / "data" / "processed" / "train.jsonl"
VALIDATION_FILE = ML_DIR / "data" / "processed" / "validation.jsonl"
TEST_FILE = ML_DIR / "data" / "processed" / "test.jsonl"

OUTPUT_DIR = ML_DIR / "models" / "intent"

MAX_LENGTH = 128


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_intents():
    path = ML_DIR / "config" / "intents.json"

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class IntentDataset(torch.utils.data.Dataset):

    def __init__(self, records, tokenizer, label_to_id):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        encoding = self.tokenizer(
            record["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        labels = np.zeros(
            len(self.label_to_id),
            dtype=np.float32,
        )

        for intent in record["intents"]:
            labels[self.label_to_id[intent]] = 1.0

        item = {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(labels)

        return item


class IntentTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        loss_function = BCEWithLogitsLoss()

        loss = loss_function(
            outputs.logits,
            labels,
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    probabilities = 1 / (
        1 + np.exp(-logits)
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "f1_micro": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def main():

    print("Loading intent dataset...")

    intents = load_intents()

    label_to_id = {
        label: index
        for index, label in enumerate(intents)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(intents),
        problem_type="multi_label_classification",
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_records = load_jsonl(TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)
    test_records = load_jsonl(TEST_FILE)

    train_dataset = IntentDataset(
        train_records,
        tokenizer,
        label_to_id,
    )

    validation_dataset = IntentDataset(
        validation_records,
        tokenizer,
        label_to_id,
    )

    test_dataset = IntentDataset(
        test_records,
        tokenizer,
        label_to_id,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=10,
        seed=42,
        report_to="none",
    )

    trainer = IntentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training intent model...")

    trainer.train()

    print("Evaluating intent model...")

    metrics = trainer.evaluate(
        test_dataset
    )

    print(metrics)

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    print(f"Intent model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()