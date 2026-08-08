import json
import os


os.makedirs(
    "dataset/intent",
    exist_ok=True
)

os.makedirs(
    "dataset/ner",
    exist_ok=True
)


samples = []


with open(
    "dataset/master.jsonl",
    encoding="utf-8"
) as f:

    for line in f:
        samples.append(
            json.loads(line)
        )


#
# Intent classification
#

with open(
    "dataset/intent/train.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        [
            {
                "text": x["text"],
                "label": x["intent"]
            }
            for x in samples
        ],
        f,
        ensure_ascii=False,
        indent=2
    )



#
# NER BIO
#

def create_bio(sample):

    text = sample["text"]

    tokens = text.split()

    labels = [
        "O"
        for _ in tokens
    ]


    entities = sample["entities"]


    for key,value in entities.items():

        if not value:
            continue


        value_tokens = value.split()


        for i in range(len(tokens)):

            if tokens[i:i+len(value_tokens)] == value_tokens:

                labels[i] = f"B-{key.upper()}"

                for j in range(
                    1,
                    len(value_tokens)
                ):
                    labels[i+j] = (
                        f"I-{key.upper()}"
                    )


    return list(
        zip(tokens, labels)
    )



with open(
    "dataset/ner/train.txt",
    "w",
    encoding="utf-8"
) as f:

    for sample in samples:

        for token,label in create_bio(sample):

            f.write(
                f"{token} {label}\n"
            )

        f.write("\n")


print("Training files generated")