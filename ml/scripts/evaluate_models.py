import json
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    classification_report as sklearn_classification_report,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


ML_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = ML_DIR / "data" / "processed" / "test.jsonl"

INTENT_MODEL = ML_DIR / "models" / "intent"
NER_MODEL = ML_DIR / "models" / "ner"


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def evaluate_intent(records):

    print()
    print("=" * 60)
    print("INTENT EVALUATION")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        str(INTENT_MODEL)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(INTENT_MODEL)
    )

    model.eval()

    label_to_id = model.config.label2id
    id_to_label = model.config.id2label

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for record in records:

            encoding = tokenizer(
                record["text"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )

            outputs = model(**encoding)

            probabilities = torch.sigmoid(
                outputs.logits
            )[0]

            predicted_ids = (
                probabilities >= 0.5
            ).nonzero(
                as_tuple=True
            )[0].tolist()

            predicted = set(
                id_to_label[i]
                for i in predicted_ids
            )

            target = set(
                record["intents"]
            )

            all_predictions.append(
                predicted
            )

            all_targets.append(
                target
            )

    labels = list(label_to_id.keys())

    y_true = []
    y_pred = []

    for target, prediction in zip(
        all_targets,
        all_predictions,
    ):

        y_true.append([
            int(label in target)
            for label in labels
        ])

        y_pred.append([
            int(label in prediction)
            for label in labels
        ])

    print()

    print(
        sklearn_classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
        )
    )

    print("Per-example predictions:")
    print()

    for record, target, prediction in zip(
        records,
        all_targets,
        all_predictions,
    ):

        if target != prediction:

            print(
                f"TEXT: {record['text']}"
            )

            print(
                f"EXPECTED: {sorted(target)}"
            )

            print(
                f"PREDICTED: {sorted(prediction)}"
            )

            print()


def evaluate_ner(records):

    print()
    print("=" * 60)
    print("NER EVALUATION")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        str(NER_MODEL)
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(NER_MODEL)
    )

    model.eval()

    id_to_label = model.config.id2label

    true_sequences = []
    predicted_sequences = []

    with torch.no_grad():

        for record in records:

            encoding = tokenizer(
                record["tokens"],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
            )

            outputs = model(**encoding)

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )[0]

            word_ids = encoding.word_ids(
                batch_index=0
            )

            true_tags = []
            predicted_tags = []

            previous_word_id = None

            for token_index, word_id in enumerate(
                word_ids
            ):

                if word_id is None:
                    continue

                if word_id == previous_word_id:
                    continue

                true_tags.append(
                    record["ner_tags"][word_id]
                )

                predicted_tags.append(
                    id_to_label[
                        predictions[token_index].item()
                    ]
                )

                previous_word_id = word_id

            true_sequences.append(true_tags)
            predicted_sequences.append(
                predicted_tags
            )

    print()

    print(
        classification_report(
            true_sequences,
            predicted_sequences,
            zero_division=0,
        )
    )

    print(
        f"Entity Precision: "
        f"{precision_score(true_sequences, predicted_sequences):.4f}"
    )

    print(
        f"Entity Recall: "
        f"{recall_score(true_sequences, predicted_sequences):.4f}"
    )

    print(
        f"Entity F1: "
        f"{f1_score(true_sequences, predicted_sequences):.4f}"
    )

    print()
    print("NER errors:")
    print()

    for record, true_tags, predicted_tags in zip(
        records,
        true_sequences,
        predicted_sequences,
    ):

        if true_tags != predicted_tags:

            print(
                f"TEXT: {record['text']}"
            )

            print(
                "TOKENS:"
            )

            for token, true_tag, predicted_tag in zip(
                record["tokens"],
                true_tags,
                predicted_tags,
            ):

                if true_tag != predicted_tag:

                    print(
                        f"  {token}: "
                        f"{true_tag} -> "
                        f"{predicted_tag}"
                    )

            print()


def main():

    records = load_jsonl(TEST_FILE)

    print(
        f"Evaluating {len(records)} test examples."
    )

    evaluate_intent(records)
    evaluate_ner(records)


if __name__ == "__main__":
    main()