import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "ml" / "data" / "raw" / "master.jsonl"

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    matches = 0

    for line_number, record in enumerate(records, start=1):

        text = record["text"]

        if "عـ" not in text:
            continue

        matches += 1

        print("\n" + "=" * 80)
        print(f"LINE: {line_number}")
        print(f"TEXT: {text}")
        print(f"INTENTS: {record['intents']}")

        print("\nTOKENS:")
        for token, tag in zip(
            record["tokens"],
            record["ner_tags"]
        ):
            if tag != "O":
                print(f"  {token}: {tag}")

    print("\n" + "=" * 80)
    print(f"Found {matches} examples containing عـ")


if __name__ == "__main__":
    main()