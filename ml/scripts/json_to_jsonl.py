import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSON array to JSONL."
    )
    parser.add_argument("input", help="Path to the input JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path.with_suffix(".jsonl")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain an array of records.")

    with output_path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(data)} records")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()