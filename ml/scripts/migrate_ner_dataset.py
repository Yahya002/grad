# this is a one-time script for unifying the old datasets
import argparse
import json
from pathlib import Path


def load_intents(path):
    intents = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            utterance = record["utterance"]

            if utterance in intents:
                raise ValueError(
                    f"Duplicate utterance in intent dataset "
                    f"at line {line_number}: {utterance!r}"
                )

            intents[utterance] = record.get("intents", [])

    return intents


def migrate(intent_path, ner_path, output_path):
    intents = load_intents(intent_path)

    converted = 0
    missing_intents = []

    with ner_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as target:

        for line_number, line in enumerate(source, start=1):

            if not line.strip():
                continue

            ner_record = json.loads(line)

            utterance = ner_record["utterance"]

            if utterance not in intents:
                missing_intents.append(
                    (line_number, utterance)
                )
                continue

            record = {
                "text": utterance,
                "intents": intents[utterance],
                "tokens": ner_record["tokens"],
                "ner_tags": ner_record["tags"],
            }

            target.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            converted += 1

    print(f"Migrated: {converted}")
    print(f"Missing intents: {len(missing_intents)}")

    if missing_intents:
        print("\nNER examples without intent annotations:")

        for line_number, utterance in missing_intents:
            print(f"  line {line_number}: {utterance}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge intent and NER datasets into master.jsonl."
    )

    parser.add_argument(
        "--intents",
        required=True,
        type=Path,
        help="Intent JSONL file"
    )

    parser.add_argument(
        "--ner",
        required=True,
        type=Path,
        help="NER JSONL file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/master.jsonl"),
        help="Output master.jsonl path"
    )

    args = parser.parse_args()

    migrate(
        args.intents,
        args.ner,
        args.output
    )