import json
import os
import random


INPUT = "dataset/clean_master.jsonl"


os.makedirs(
    "dataset/intent",
    exist_ok=True
)

os.makedirs(
    "dataset/ner",
    exist_ok=True
)


samples = []


with open(INPUT, encoding="utf-8") as f:

    for line in f:
        samples.append(
            json.loads(line)
        )


random.shuffle(samples)


split = int(len(samples) * 0.8)

train = samples[:split]
val = samples[split:]


# ==========================
# Intent dataset
# ==========================

def build_intent(data):

    return [
        {
            "text": item["text"],
            "label": item["intent"]
        }
        for item in data
    ]


with open(
    "dataset/intent/train.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        build_intent(train),
        f,
        ensure_ascii=False,
        indent=2
    )


with open(
    "dataset/intent/validation.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        build_intent(val),
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================
# NER dataset
# ==========================


def tokenize(text):
    return text.split()



def create_bio(item):

    tokens = tokenize(
        item["text"]
    )

    labels = [
        "O"
        for _ in tokens
    ]


    for entity_type, value in item["entities"].items():

        if not value:
            continue


        entity_tokens = value.split()


        for i in range(
            len(tokens)
        ):

            window = tokens[
                i:i+len(entity_tokens)
            ]


            if window == entity_tokens:

                labels[i] = (
                    "B-"
                    + entity_type.upper()
                )


                for j in range(
                    1,
                    len(entity_tokens)
                ):

                    labels[i+j] = (
                        "I-"
                        + entity_type.upper()
                    )


    return list(
        zip(tokens, labels)
    )



def write_ner(data, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:

            for token,label in create_bio(item):

                f.write(
                    f"{token} {label}\n"
                )

            f.write("\n")



write_ner(
    train,
    "dataset/ner/train.txt"
)


write_ner(
    val,
    "dataset/ner/validation.txt"
)


print(
    "Generated datasets:"
)

print(
    "Intent:",
    len(train),
    "train",
    len(val),
    "validation"
)

print(
    "NER:",
    len(train),
    "train",
    len(val),
    "validation"
)