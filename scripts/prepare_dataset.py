import json
from collections import Counter


INPUT = "dataset/master.jsonl"
OUTPUT = "dataset/clean_master.jsonl"


def clean(value):

    if value in [None, "", "null", "None"]:
        return None

    return value


seen = set()
cleaned = []


with open(INPUT, encoding="utf-8") as f:

    for line in f:

        item = json.loads(line)

        text = item["text"].strip()

        key = (
            text,
            item["intent"]
        )

        if key in seen:
            continue

        seen.add(key)


        entities = {
            k: clean(v)
            for k,v in item["entities"].items()
        }


        cleaned.append({
            "text": text,
            "intent": item["intent"],
            "entities": entities
        })


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    for item in cleaned:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


print("Samples:", len(cleaned))

print(
    Counter(
        x["intent"]
        for x in cleaned
    )
)