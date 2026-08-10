import json
import random
from pathlib import Path


ML_DIR = Path(__file__).resolve().parent.parent

INPUT = ML_DIR / "data" / "raw" / "master.jsonl"
OUTPUT = ML_DIR / "data" / "processed"

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


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


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_dataset()

    random.seed(SEED)
    random.shuffle(records)

    total = len(records)

    train_end = int(total * TRAIN_RATIO)

    validation_end = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = records[:train_end]

    validation = records[
        train_end:validation_end
    ]

    test = records[
        validation_end:
    ]

    save_dataset(train, "train.jsonl")
    save_dataset(validation, "validation.jsonl")
    save_dataset(test, "test.jsonl")

    print(f"Total:      {len(records)}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")


if __name__ == "__main__":
    main()