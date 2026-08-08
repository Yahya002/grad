from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./models/intent"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

id2label = model.config.id2label


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return id2label[prediction]


tests = [
    "مرحبا",
    "شلونك",
    "بدي تكسي من كافيه راية للمشفى الرازي",
    "يرجى الغاء الطلب",
    "اريد توصيل طرد"
]


for t in tests:
    print(t, "=>", predict(t))