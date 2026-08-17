import json
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


ML_DIR = Path(__file__).resolve().parent.parent

INPUT = ML_DIR / "data" / "raw" / "generated_taxi.jsonl"
OUTPUT = ML_DIR / "data" / "processed"
INTENTS_FILE = ML_DIR / "config" / "intents.json"

VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    records = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_dataset(records, filename):
    path = OUTPUT / filename

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def build_intent_matrix(records, intents):
    intent_to_index = {
        intent: index
        for index, intent in enumerate(intents)
    }

    matrix = []

    for record in records:
        row = [0] * len(intents)

        for intent in record["intents"]:
            row[intent_to_index[intent]] = 1

        matrix.append(row)

    return matrix


def print_distribution(name, records, intents):
    counts = {
        intent: 0
        for intent in intents
    }

    for record in records:
        for intent in record["intents"]:
            counts[intent] += 1

    print(f"\n{name}: {len(records)} examples")

    for intent in intents:
        print(
            f"  {intent:<20} {counts[intent]}"
        )


def split_dataset(records, intents):
    labels = build_intent_matrix(
        records,
        intents
    )

    # 90% train+validation
    # 10% test

    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=SEED,
    )

    train_val_indices, test_indices = next(
        first_splitter.split(
            records,
            labels
        )
    )

    train_val = [
        records[i]
        for i in train_val_indices
    ]

    test = [
        records[i]
        for i in test_indices
    ]

    train_val_labels = [
        labels[i]
        for i in train_val_indices
    ]

    # Of the remaining 90%, validation should be:
    #
    # 0.1 / 0.9 = 1/9
    #
    # This produces approximately:
    #
    # train      80%
    # validation 10%
    # test       10%

    validation_fraction = (
        VALIDATION_RATIO
        / (1.0 - TEST_RATIO)
    )

    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=SEED,
    )

    train_indices, validation_indices = next(
        second_splitter.split(
            train_val,
            train_val_labels
        )
    )

    train = [
        train_val[i]
        for i in train_indices
    ]

    validation = [
        train_val[i]
        for i in validation_indices
    ]

    return train, validation, test


def main():
    records = load_dataset()
    intents = load_json(INTENTS_FILE)

    train, validation, test = split_dataset(
        records,
        intents
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    save_dataset(
        train,
        "train.jsonl"
    )

    save_dataset(
        validation,
        "validation.jsonl"
    )

    save_dataset(
        test,
        "test.jsonl"
    )

    print(f"Total:      {len(records)}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print_distribution(
        "TRAIN",
        train,
        intents
    )

    print_distribution(
        "VALIDATION",
        validation,
        intents
    )

    print_distribution(
        "TEST",
        test,
        intents
    )


if __name__ == "__main__":
    main()