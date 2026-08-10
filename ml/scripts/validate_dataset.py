import json
import sys
from pathlib import Path

INTENTS_FILE = Path("config/intents.json")
ENTITIES_FILE = Path("config/entities.json")
DATASET_FILE = Path("data/raw/master.jsonl")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate():

    allowed_intents = set(load_json(INTENTS_FILE))
    allowed_entities = set(load_json(ENTITIES_FILE))

    allowed_ner_tags = {"O"}

    for entity in allowed_entities:
        allowed_ner_tags.add(f"B-{entity}")
        allowed_ner_tags.add(f"I-{entity}")

    errors = []
    examples = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            examples += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(
                    f"Line {line_number}: invalid JSON: {e}"
                )
                continue

            # Required fields
            required = {
                "text",
                "intents",
                "tokens",
                "ner_tags",
            }

            missing = required - item.keys()

            if missing:
                errors.append(
                    f"Line {line_number}: missing fields: {missing}"
                )
                continue

            text = item["text"]
            intents = item["intents"]
            tokens = item["tokens"]
            ner_tags = item["ner_tags"]

            # Basic types
            if not isinstance(text, str):
                errors.append(
                    f"Line {line_number}: text must be string"
                )

            if not isinstance(intents, list):
                errors.append(
                    f"Line {line_number}: intents must be array"
                )

            if not isinstance(tokens, list):
                errors.append(
                    f"Line {line_number}: tokens must be array"
                )

            if not isinstance(ner_tags, list):
                errors.append(
                    f"Line {line_number}: ner_tags must be array"
                )

            if not (
                isinstance(intents, list)
                and isinstance(tokens, list)
                and isinstance(ner_tags, list)
            ):
                continue

            # Intent validation
            unknown_intents = set(intents) - allowed_intents

            if unknown_intents:
                errors.append(
                    f"Line {line_number}: "
                    f"unknown intents: {unknown_intents}"
                )

            # Duplicate intents
            if len(intents) != len(set(intents)):
                errors.append(
                    f"Line {line_number}: duplicate intents"
                )

            # Token/tag alignment
            if len(tokens) != len(ner_tags):
                errors.append(
                    f"Line {line_number}: "
                    f"{len(tokens)} tokens but "
                    f"{len(ner_tags)} NER tags"
                )

            # NER labels
            for tag in ner_tags:

                if tag not in allowed_ner_tags:
                    errors.append(
                        f"Line {line_number}: "
                        f"unknown NER tag: {tag}"
                    )

    print(f"Validated examples: {examples}")

    if errors:
        print(f"\nFound {len(errors)} errors:\n")

        for error in errors[:100]:
            print(error)

        if len(errors) > 100:
            print(
                f"\n...and {len(errors) - 100} more."
            )

        sys.exit(1)

    print("Dataset is valid.")


if __name__ == "__main__":
    validate()