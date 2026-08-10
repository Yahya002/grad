import json
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


ML_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

TRAIN_FILE = ML_DIR / "data" / "processed" / "train.jsonl"
VALIDATION_FILE = ML_DIR / "data" / "processed" / "validation.jsonl"
TEST_FILE = ML_DIR / "data" / "processed" / "test.jsonl"

OUTPUT_DIR = ML_DIR / "models" / "ner"


def load_jsonl(path):

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_labels():

    entities_path = ML_DIR / "config" / "entities.json"

    with entities_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        entities = json.load(f)

    labels = ["O"]

    for entity in entities:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")

    return labels


def tokenize_and_align_labels(
    example,
    tokenizer,
    label_to_id,
):

    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    word_ids = tokenized.word_ids()

    labels = []

    previous_word_id = None

    for word_id in word_ids:

        if word_id is None:
            labels.append(-100)

        elif word_id != previous_word_id:
            tag = example["ner_tags"][word_id]

            labels.append(
                label_to_id[tag]
            )

        else:
            tag = example["ner_tags"][word_id]

            # For subword pieces, keep the
            # same entity type.

            if tag.startswith("B-"):
                tag = "I-" + tag[2:]

            labels.append(
                label_to_id[tag]
            )

        previous_word_id = word_id

    tokenized["labels"] = labels

    return tokenized


def prepare_dataset(
    records,
    tokenizer,
    label_to_id,
):

    dataset = Dataset.from_list(records)

    return dataset.map(
        lambda example:
            tokenize_and_align_labels(
                example,
                tokenizer,
                label_to_id,
            ),
        remove_columns=dataset.column_names,
    )


def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2,
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels,
    ):

        for pred, actual in zip(
            prediction,
            label,
        ):

            if actual != -100:
                true_predictions.append(pred)
                true_labels.append(actual)

    accuracy = np.mean(
        np.array(true_predictions)
        == np.array(true_labels)
    )

    return {
        "accuracy": accuracy,
    }


def main():

    print("Loading NER dataset...")

    labels = load_labels()

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_records = load_jsonl(TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)
    test_records = load_jsonl(TEST_FILE)

    train_dataset = prepare_dataset(
        train_records,
        tokenizer,
        label_to_id,
    )

    validation_dataset = prepare_dataset(
        validation_records,
        tokenizer,
        label_to_id,
    )

    test_dataset = prepare_dataset(
        test_records,
        tokenizer,
        label_to_id,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training NER model...")

    trainer.train()

    print("Evaluating NER model...")

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

    print(f"NER model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()