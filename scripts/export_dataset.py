from pymongo import MongoClient
import json
from datetime import datetime


client = MongoClient("mongodb://localhost:27017")

db = client.taxi_bot


logs = db.llm_logs.find({
    "validated": True,
    "corrected": {
        "$exists": True
    }
})


dataset = []


for log in logs:

    corrected = log["corrected"]

    sample = {
        "text": log["message"],

        "intent": corrected.get("intent"),

        "entities": {
            "name": corrected.get("name"),
            "phone": corrected.get("phone"),
            "from": corrected.get("from"),
            "to": corrected.get("to")
        },

        "metadata": {
            "source": "human_annotation",
            "timestamp": str(
                log.get(
                    "timestamp",
                    datetime.now()
                )
            )
        }
    }

    dataset.append(sample)



with open(
    "dataset/master.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for item in dataset:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


print(
    f"Exported {len(dataset)} samples"
)