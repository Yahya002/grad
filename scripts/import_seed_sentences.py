from pathlib import Path

from app.services.intent_service import analyze_message
from app.models.llm_log_model import save_llm_log
from flask import Flask

from app.database.mongo import init_db
from app.models.llm_log_model import save_llm_log

app = Flask(__name__)
init_db(app)

def load_seed_dataset(path):
    current_intent = None
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # section header
            if line.startswith("[") and line.endswith("]"):
                current_intent = line[1:-1]
                continue

            if current_intent is None:
                raise ValueError(f"Sentence outside any section: {line}")

            samples.append({
                "intent": current_intent,
                "text": line
            })

    return samples


def main():

    samples = load_seed_dataset("data/seed_01.txt")

    print(f"Loaded {len(samples)} samples.\n")

    for sample in samples:

        prediction = analyze_message(sample["text"])

        save_llm_log(
            user_id=0,
            message=sample["text"],
            response=prediction,
            expected={
                "intent": sample["intent"]
            },
            source="seed"
        )

        predicted = prediction.get("intent", "unknown")

        status = "✓" if predicted == sample["intent"] else "✗"

        print(
            f"{status} "
            f"expected={sample['intent']:<14} "
            f"predicted={predicted:<14} "
            f"{sample['text']}"
        )

    print(f"\nImported {len(samples)} samples.")


if __name__ == "__main__":
    main()