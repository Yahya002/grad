from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
ML_DIR = BASE_DIR / "ml"

INTENT_MODEL_DIR = ML_DIR / "models" / "intent"
NER_MODEL_DIR = ML_DIR / "models" / "ner"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"ML device: {DEVICE}")


# ============================================================
# GLOBAL MODELS
# ============================================================

_intent_model = None
_intent_tokenizer = None

_ner_model = None
_ner_tokenizer = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    global _intent_model
    global _intent_tokenizer
    global _ner_model
    global _ner_tokenizer

    if _intent_model is None:

        print("Loading intent model...")

        _intent_tokenizer = AutoTokenizer.from_pretrained(
            str(INTENT_MODEL_DIR)
        )

        _intent_model = AutoModelForSequenceClassification.from_pretrained(
            str(INTENT_MODEL_DIR)
        )

        _intent_model.to(DEVICE)
        _intent_model.eval()

        print("Intent model loaded.")

    if _ner_model is None:

        print("Loading NER model...")

        _ner_tokenizer = AutoTokenizer.from_pretrained(
            str(NER_MODEL_DIR)
        )

        _ner_model = AutoModelForTokenClassification.from_pretrained(
            str(NER_MODEL_DIR)
        )

        _ner_model.to(DEVICE)
        _ner_model.eval()

        print("NER model loaded.")

    return (
        _intent_model,
        _intent_tokenizer,
        _ner_model,
        _ner_tokenizer,
    )


# ============================================================
# INTENT
# ============================================================

def predict_intents(message):

    model, tokenizer, _, _ = load_models()

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

    intents = []

    for index, probability in enumerate(probabilities):

        if probability.item() >= 0.5:

            label = model.config.id2label[index]

            intents.append(label)

    if not intents:
        intents = ["unknown"]

    return intents


# ============================================================
# NER
# ============================================================

def predict_entities(message):

    _, _, model, tokenizer = load_models()

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")

    word_ids = inputs.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **model_inputs
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )[0]

    # ========================================================
    # GROUP SUBWORDS BY ORIGINAL WORD
    # ========================================================

    words = {}

    for token_index, word_id in enumerate(word_ids):

        if word_id is None:
            continue

        start = offset_mapping[0][token_index][0].item()
        end = offset_mapping[0][token_index][1].item()

        if start == end:
            continue

        label_id = predictions[token_index].item()
        label = model.config.id2label[label_id]

        if word_id not in words:

            words[word_id] = {
                "start": start,
                "end": end,
                "labels": [],
            }

        words[word_id]["start"] = min(
            words[word_id]["start"],
            start,
        )

        words[word_id]["end"] = max(
            words[word_id]["end"],
            end,
        )

        words[word_id]["labels"].append(label)

    # ========================================================
    # ONE LABEL PER ORIGINAL WORD
    # ========================================================

    word_predictions = []

    for word_id in sorted(words):

        word = words[word_id]

        labels = word["labels"]

        # Prefer B/I over O if any subword is classified
        # as part of an entity.
        entity_labels = [
            label
            for label in labels
            if label.startswith("B-")
            or label.startswith("I-")
        ]

        if entity_labels:

            # Prefer B- if available.
            label = next(
                (
                    x
                    for x in entity_labels
                    if x.startswith("B-")
                ),
                entity_labels[0],
            )

        else:
            label = "O"

        word_predictions.append({
            "start": word["start"],
            "end": word["end"],
            "label": label,
        })

    # ========================================================
    # BUILD ENTITIES
    # ========================================================

    entities = []
    current_entity = None

    for word in word_predictions:

        label = word["label"]

        # --------------------------------------------
        # Begin entity
        # --------------------------------------------

        if label.startswith("B-"):

            if current_entity is not None:
                entities.append(current_entity)

            current_entity = {
                "type": label[2:],
                "start": word["start"],
                "end": word["end"],
            }

        # --------------------------------------------
        # Continue entity
        # --------------------------------------------

        elif (
            label.startswith("I-")
            and current_entity is not None
            and current_entity["type"] == label[2:]
        ):

            current_entity["end"] = word["end"]

        # --------------------------------------------
        # Outside / broken entity
        # --------------------------------------------

        else:

            if current_entity is not None:
                entities.append(current_entity)

            current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    # ========================================================
    # RETURN TEXT
    # ========================================================

    return [
        {
            "type": entity["type"],
            "text": message[
                entity["start"]:entity["end"]
            ],
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in entities
    ]

def detect_location_role(message, entity):

    text = entity["text"]
    start = entity["start"]

    # ========================================================
    # Connected destination prefixes INSIDE the NER entity
    #
    # لدوار
    # لمشفى
    # لعند...
    # عالمشروع
    # ========================================================

    if text.startswith("ل"):
        return "dest"

    if text.startswith("ع"):
        return "dest"

    # ========================================================
    # Text immediately before entity
    # ========================================================

    before = message[:start].rstrip()

    # ========================================================
    # Pickup markers
    # ========================================================

    pickup_markers = [
        "من عند",
        "من قدام",
        "من جنب",
        "من",
    ]

    for marker in pickup_markers:

        if before.endswith(marker):
            return "pickup"

    # ========================================================
    # Destination markers
    # ========================================================

    destination_markers = [
        "لعند",
        "إلى",
        "الى",
        "ل",
        "ع",
    ]

    for marker in destination_markers:

        if before.endswith(marker):
            return "dest"

    return None

def normalize_location(text, role):

    text = text.strip()

    # ========================================================
    # Destination connected prefixes
    #
    # l + location
    #
    # لدوار الساعة -> دوار الساعة
    # لمشفى الحكمة -> مشفى الحكمة
    # لعند الجامعة -> عند الجامعة
    #
    # ========================================================

    if role == "dest":

        if text.startswith("لعند"):
            return text[2:]

        if text.startswith("ل"):
            return text[1:]

        if text.startswith("ع"):
            return text[1:]

    return text

# ============================================================
# LOCATION RESOLVER
# ============================================================

def resolve_locations(message, entities, session_status):

    result = {
        "from": None,
        "to": None,
    }

    locations = [
        entity
        for entity in entities
        if entity["type"] == "LOCATION"
    ]

    if not locations:
        return result

    # ========================================================
    # 1. SESSION CONTEXT
    # ========================================================

    if session_status == "waiting_for_pickup":

        result["from"] = normalize_location(
            locations[0]["text"],
            "pickup",
        )

        return result

    if session_status == "waiting_for_dest":

        result["to"] = normalize_location(
            locations[0]["text"],
            "dest",
        )

        return result

    # ========================================================
    # 2. TWO LOCATION REQUEST
    # ========================================================

    if len(locations) >= 2:

        first = locations[0]
        second = locations[1]

        first_text = first["text"]
        second_text = second["text"]

        # --------------------------------------------
        # First location
        # --------------------------------------------

        first_role = detect_location_role(
            message,
            first,
        )

        # --------------------------------------------
        # Second location
        # --------------------------------------------

        second_role = detect_location_role(
            message,
            second,
        )

        # --------------------------------------------
        # Explicit:
        #
        # من X لـ Y
        # من X لY
        # من X لعند Y
        # من X إلى Y
        # --------------------------------------------

        if (
            first_role == "pickup"
            and second_role == "dest"
        ):

            result["from"] = normalize_location(
                first_text,
                "pickup",
            )

            result["to"] = normalize_location(
                second_text,
                "dest",
            )

            return result

        # --------------------------------------------
        # Destination first:
        #
        # لـ Y من X
        # لعند Y من X
        # إلى Y من X
        # --------------------------------------------

        if (
            first_role == "dest"
            and second_role == "pickup"
        ):

            result["from"] = normalize_location(
                second_text,
                "pickup",
            )

            result["to"] = normalize_location(
                first_text,
                "dest",
            )

            return result

        # --------------------------------------------
        # If only one side was explicitly identified
        # --------------------------------------------

        if first_role == "pickup":

            result["from"] = normalize_location(
                first_text,
                "pickup",
            )

            result["to"] = normalize_location(
                second_text,
                "dest",
            )

            return result

        if second_role == "dest":

            result["from"] = normalize_location(
                first_text,
                "pickup",
            )

            result["to"] = normalize_location(
                second_text,
                "dest",
            )

            return result

        # --------------------------------------------
        # Safe taxi-request fallback:
        #
        # first location = pickup
        # second location = destination
        # --------------------------------------------

        result["from"] = normalize_location(
            first_text,
            "pickup",
        )

        result["to"] = normalize_location(
            second_text,
            "dest",
        )

        return result

    # ========================================================
    # 3. SINGLE LOCATION
    # ========================================================

    entity = locations[0]

    role = detect_location_role(
        message,
        entity,
    )

    if role == "pickup":

        result["from"] = normalize_location(
            entity["text"],
            "pickup",
        )

    elif role == "dest":

        result["to"] = normalize_location(
            entity["text"],
            "dest",
        )

    return result

# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict(message, session_status="idle"):

    intents = predict_intents(message)
    entities = predict_entities(message)

    print("NER ENTITIES:", entities)

    resolved = resolve_locations(
        message,
        entities,
        session_status,
    )

    return {
        "intents": intents,
        "from": resolved["from"],
        "to": resolved["to"],
    }

def has_pickup_marker(message, start):
    """
    Check whether a LOCATION beginning at `start` is preceded
    by a pickup expression.
    """

    before = message[:start].rstrip()

    markers = [
        "من عند",
        "من قدام",
        "من جنب",
        "من",
    ]

    return any(
        before.endswith(marker)
        for marker in markers
    )


def has_destination_marker(message, start):
    """
    Check whether a LOCATION beginning at `start` is preceded
    by a destination expression.

    Handles both separated and connected Syrian Arabic forms:

        ل مشفى
        لمشفى
        لعند مشفى
        لعندمشفى
        إلى مشفى
        ع المشروع
        عالمشروع
    """

    before = message[:start].rstrip()

    separated_markers = [
        "لعند",
        "إلى",
        "الى",
        "ل",
        "ع",
    ]

    if any(
        before.endswith(marker)
        for marker in separated_markers
    ):
        return True

    # Connected prefix.

    if start > 0:

        previous_char = message[start - 1]

        if previous_char in {"ل", "ع"}:
            return True

    return False

# ============================================================
# BACKWARD-COMPATIBLE ENTRY POINT
# ============================================================

def call_llm(message, session_status="idle"):
    return predict(message, session_status)