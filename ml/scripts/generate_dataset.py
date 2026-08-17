import json
import random
import re
from pathlib import Path

# ============================================================
# Taxi language-component dataset generator
#
# Surface text stays natural Syrian Arabic.
#
# Example:
#   text:   بدي تكسي لمشفى الحكمة
#   tokens: بدي | تكسي | ل | مشفى | الحكمة
#   tags:   O   | O    | O | B-LOCATION | I-LOCATION
#
# LOCATION is the only location NER label.
# PICKUP/DEST are semantic roles used by the generator only.
# The runtime state machine decides whether a LOCATION is pickup
# or destination.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

COMPONENTS_DIR = BASE_DIR / "data" / "language-components"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "generated_taxi.jsonl"

GREETINGS_FILE = COMPONENTS_DIR / "greeting.json"
PLEASE_FILE = COMPONENTS_DIR / "please.json"
THANKS_FILE = COMPONENTS_DIR / "thanks.json"
LOCATIONS_FILE = COMPONENTS_DIR / "locations.json"

SEED = 42
random.seed(SEED)

# Keep generation finite. Increase these after inspecting quality.
MAX_COMPLETE = 30000
MAX_PICKUP_ONLY = 8000
MAX_DEST_ONLY = 8000

# Standalone language components.
MAX_STANDALONE_GREETING = 1000
MAX_STANDALONE_PLEASE = 1000
MAX_STANDALONE_THANKS = 1000


REQUEST_PHRASES = [
    "بدي تكسي",
    "بدي سيارة",
    "بدي حدا يوصلني",
    "ممكن تكسي",
    "ممكن سيارة",
    "ممكن تحجزولي تكسي",
    "دبرولي تكسي",
    "دبرولي سيارة",
    "تأمنولي تكسي",
    "تأمنولي سيارة",
    "بدي تكسي تجي",
    "بدي سيارة تجي",
    "ممكن تبعتولي تكسي",
    "ممكن تأمنولي تكسي",
]


# ------------------------------------------------------------
# Complete request templates
#
# {PICKUP} = a location occupying pickup position
# {DEST}   = a location occupying destination position
#
# {DEST_L} means destination with connected ل:
#   لمشفى الحكمة
#
# {DEST_B} means destination with connected ب:
#   بالمشروع السادس
#
# {DEST_A} means destination with connected ع:
#   عالمشروع السادس
#
# The same suffixes are available for PICKUP.
# ------------------------------------------------------------

COMPLETE_TEMPLATES = [
    # Pickup first
    "{REQUEST} من {PICKUP} لـDEST",
    "{REQUEST} من {PICKUP} {DEST_L}",
    "{REQUEST} من {PICKUP} لعند {DEST}",
    "{REQUEST} من {PICKUP} إلى {DEST}",
    "{REQUEST} من عند {PICKUP} {DEST_L}",
    "{REQUEST} من قدام {PICKUP} {DEST_L}",
    "{REQUEST} من جنب {PICKUP} {DEST_L}",
    "{REQUEST} {PICKUP_A} {DEST_L}",
    "{REQUEST} {PICKUP_B} {DEST_L}",
    "{REQUEST} ياخدني من {PICKUP} {DEST_L}",
    "{REQUEST} ياخدني من {PICKUP} لعند {DEST}",
    "{REQUEST} يوديني من {PICKUP} {DEST_L}",
    "{REQUEST} يوصلني من {PICKUP} {DEST_L}",
    "{REQUEST} تجي {PICKUP_A} وتوصلني {DEST_L}",
    "{REQUEST} تجي {PICKUP_B} وتوصلني {DEST_L}",

    # Destination first
    "{REQUEST} {DEST_L} من {PICKUP}",
    "{REQUEST} لعند {DEST} من {PICKUP}",
    "{REQUEST} إلى {DEST} من {PICKUP}",
    "{REQUEST} {DEST_A} من {PICKUP}",
    "{REQUEST} {DEST_L}، من عند {PICKUP}",
    "{REQUEST} لعند {DEST}، من عند {PICKUP}",
    "{REQUEST} وصلني {DEST_L} من {PICKUP}",
    "{REQUEST} وديني {DEST_L} من {PICKUP}",
    "{REQUEST} يوصلني {DEST_L} من {PICKUP}",
]

# Remove accidental literal marker from the first template while
# keeping the intended meaning explicit.
COMPLETE_TEMPLATES = [
    t.replace(" لـDEST", " {DEST_L}")
    for t in COMPLETE_TEMPLATES
]


PICKUP_ONLY_TEMPLATES = [
    "{REQUEST} من {PICKUP}",
    "{REQUEST} من عند {PICKUP}",
    "{REQUEST} من قدام {PICKUP}",
    "{REQUEST} من جنب {PICKUP}",
    "{REQUEST} {PICKUP_A}",
    "{REQUEST} {PICKUP_B}",
    "{REQUEST} تجي {PICKUP_A}",
    "{REQUEST} تجي {PICKUP_B}",
    "{REQUEST} ياخدني من {PICKUP}",
]


DEST_ONLY_TEMPLATES = [
    "{REQUEST} {DEST_L}",
    "{REQUEST} لعند {DEST}",
    "{REQUEST} إلى {DEST}",
    "{REQUEST} {DEST_A}",
    "{REQUEST} وصلني {DEST_L}",
    "{REQUEST} وديني {DEST_L}",
    "{REQUEST} يوصلني {DEST_L}",
]


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_list(values):
    return list(dict.fromkeys(
        x.strip()
        for x in values
        if isinstance(x, str) and x.strip()
    ))


def load_components():
    greetings = clean_list(load_json(GREETINGS_FILE))
    please = clean_list(load_json(PLEASE_FILE))
    thanks = clean_list(load_json(THANKS_FILE))

    location_data = load_json(LOCATIONS_FILE)
    if not isinstance(location_data, dict):
        raise ValueError("locations.json must contain an object of arrays")

    locations = []
    for values in location_data.values():
        if isinstance(values, list):
            locations.extend(
                x.strip()
                for x in values
                if isinstance(x, str) and x.strip()
            )

    locations = list(dict.fromkeys(locations))

    if not locations:
        raise ValueError("locations.json contains no locations")

    return greetings, please, thanks, locations


# ------------------------------------------------------------
# Prefix handling
# ------------------------------------------------------------

PREFIXES = {
    "L": "ل",
    "B": "ب",
    "A": "ع",
}


def attach_prefix(prefix, location):
    """
    Natural connected Syrian-Arabic surface form.

    ل + مشفى الحكمة  -> لمشفى الحكمة
    ل + المشروع السادس -> للمشروع السادس
    ب + المشروع السادس -> بالمشروع السادس
    ع + المشروع السادس -> عالمشروع السادس

    No spaces are introduced.
    """

    words = location.split()
    if not words:
        return location

    first = words[0]

    if first.startswith("ال"):
        first = prefix + first
    else:
        first = prefix + first

    return " ".join([first] + words[1:])


# ------------------------------------------------------------
# Rendering
# ------------------------------------------------------------

PLACEHOLDER_RE = re.compile(
    r"\{(PICKUP|DEST)(?:_(L|B|A))?\}"
)


def render_template(template, request_phrase, pickup=None, dest=None):
    template = template.replace("{REQUEST}", request_phrase)

    parts = []
    entities = []

    cursor = 0

    for match in PLACEHOLDER_RE.finditer(template):
        parts.append(template[cursor:match.start()])

        role = match.group(1)
        prefix_code = match.group(2)

        value = pickup if role == "PICKUP" else dest
        if value is None:
            raise ValueError(
                f"Template needs {role}: {template}"
            )

        prefix = PREFIXES.get(prefix_code)

        rendered = (
            attach_prefix(prefix, value)
            if prefix
            else value
        )

        parts.append(rendered)

        entities.append({
            "text": value,
            "role": role,
            "prefix": prefix,
        })

        cursor = match.end()

    parts.append(template[cursor:])

    return "".join(parts), entities


# ------------------------------------------------------------
# Surface-text tokenization
#
# This deliberately differs from text.split() when a connected
# prefix exists.
# ------------------------------------------------------------

def surface_tokens(text):
    return text.split()


def strip_edge_punctuation(token):
    return re.sub(r"^[،,:;.!؟]+|[،,:;.!؟]+$", "", token)


def prefix_split(token, prefix, first_location_word):
    """
    Split only the generated connected form.

    لمشفى -> ل + مشفى
    للمشروع -> ل + المشروع
    بالمشروع -> ب + المشروع
    عالمشروع -> ع + المشروع
    """

    clean = strip_edge_punctuation(token)
    expected = attach_prefix(prefix, first_location_word)

    if clean != expected:
        return None

    # Preserve trailing punctuation on the location part.
    suffix = token[len(clean):]
    location_part = first_location_word + suffix

    return [prefix, location_part]


def annotate(text, entities):
    raw = surface_tokens(text)

    # Find each semantic entity in order.
    # PICKUP/DEST are generator-internal roles only.
    # The actual NER label is always LOCATION.
    matches = []
    search_from = 0

    for entity in entities:
        words = entity["text"].split()
        prefix = entity["prefix"]
        role = entity["role"]

        if role not in {"PICKUP", "DEST"}:
            raise ValueError(
                f"Invalid entity role '{role}'"
            )

        found = None

        for start in range(search_from, len(raw)):
            first = strip_edge_punctuation(raw[start])

            expected_first = (
                attach_prefix(prefix, words[0])
                if prefix
                else words[0]
            )

            if first != expected_first:
                continue

            if start + len(words) > len(raw):
                continue

            valid = True

            for offset, word in enumerate(words):
                current = strip_edge_punctuation(
                    raw[start + offset]
                )

                expected = (
                    attach_prefix(prefix, word)
                    if prefix and offset == 0
                    else word
                )

                if current != expected:
                    valid = False
                    break

            if valid:
                found = list(
                    range(start, start + len(words))
                )
                break

        if found is None:
            raise ValueError(
                f"Could not align {role} '{entity['text']}' "
                f"in:\n{text}"
            )

        matches.append((entity, found))
        search_from = found[-1] + 1

    index_map = {}

    for entity, positions in matches:
        for n, pos in enumerate(positions):
            index_map[pos] = (
                entity,
                "B-LOCATION" if n == 0 else "I-LOCATION",
            )

    tokens = []
    tags = []

    for i, raw_token in enumerate(raw):

        if i not in index_map:
            tokens.append(raw_token)
            tags.append("O")
            continue

        entity, tag = index_map[i]

        # The connected prefix exists only on the first
        # token of the location.
        is_first_location_token = (
            tag == "B-LOCATION"
        )

        if entity["prefix"] and is_first_location_token:
            split = prefix_split(
                raw_token,
                entity["prefix"],
                entity["text"].split()[0],
            )

            if split is None:
                raise ValueError(
                    f"Failed to split prefix in '{raw_token}'"
                )

            # Connected preposition is outside LOCATION.
            tokens.append(split[0])
            tags.append("O")

            # Actual location word.
            tokens.append(split[1])
            tags.append(tag)

        else:
            tokens.append(raw_token)
            tags.append(tag)

    if len(tokens) != len(tags):
        raise ValueError("Token/tag length mismatch")

    return tokens, tags

# ------------------------------------------------------------
# Records
# ------------------------------------------------------------

def record(text, intents, entities):
    tokens, tags = annotate(text, entities)

    return {
        "text": text,
        "intents": intents,
        "tokens": tokens,
        "ner_tags": tags,
    }


# ------------------------------------------------------------
# Optional components
# ------------------------------------------------------------

def request_variants(text, entities, greetings, please, thanks):
    """
    Greeting can only precede the request.
    Please can precede OR follow the request.
    Thanks can follow the request.

    No greeting-after-request variants are generated.
    """

    result = []

    # Plain
    result.append(
        (text, entities, ["request_taxi"])
    )

    # Greeting before request
    if greetings:
        greeting = random.choice(greetings)
        result.append(
            (
                f"{greeting}، {text}",
                entities,
                ["request_taxi", "greeting"],
            )
        )

    # Please before request
    if please:
        polite = random.choice(please)
        result.append(
            (
                f"{polite}، {text}",
                entities,
                ["request_taxi", "politeness"],
            )
        )

    # Please after request
    if please:
        polite = random.choice(please)
        result.append(
            (
                f"{text}، {polite}",
                entities,
                ["request_taxi", "politeness"],
            )
        )

    # Greeting + please + request
    if greetings and please:
        greeting = random.choice(greetings)
        polite = random.choice(please)
        result.append(
            (
                f"{greeting}، {polite}، {text}",
                entities,
                ["request_taxi", "greeting", "politeness"],
            )
        )

    # Request + thanks
    if thanks:
        thank = random.choice(thanks)
        result.append(
            (
                f"{text}، {thank}",
                entities,
                ["request_taxi", "thanks"],
            )
        )

    # Greeting + request + thanks
    if greetings and thanks:
        greeting = random.choice(greetings)
        thank = random.choice(thanks)
        result.append(
            (
                f"{greeting}، {text}، {thank}",
                entities,
                ["request_taxi", "greeting", "thanks"],
            )
        )

    # Greeting + please + request + thanks
    if greetings and please and thanks:
        greeting = random.choice(greetings)
        polite = random.choice(please)
        thank = random.choice(thanks)
        result.append(
            (
                f"{greeting}، {polite}، {text}، {thank}",
                entities,
                ["request_taxi", "greeting", "politeness", "thanks"],
            )
        )

    # Please + request + thanks
    if please and thanks:
        polite = random.choice(please)
        thank = random.choice(thanks)
        result.append(
            (
                f"{polite}، {text}، {thank}",
                entities,
                ["request_taxi", "politeness", "thanks"],
            )
        )

    return result


# ------------------------------------------------------------
# Complete requests
# ------------------------------------------------------------

def generate_complete(
    locations,
    greetings,
    please,
    thanks,
):
    records = []

    while len(records) < MAX_COMPLETE:
        template = random.choice(COMPLETE_TEMPLATES)
        pickup, dest = random.sample(locations, 2)

        text, entities = render_template(
            template,
            random.choice(REQUEST_PHRASES),
            pickup=pickup,
            dest=dest,
        )

        variant = random.choice(
            request_variants(
                text,
                entities,
                greetings,
                please,
                thanks,
            )
        )

        text, entities, intents = variant

        records.append(
            record(text, intents, entities)
        )

    return records


# ------------------------------------------------------------
# Partial requests
# ------------------------------------------------------------

def generate_pickup_only(locations):
    records = []

    while len(records) < MAX_PICKUP_ONLY:
        template = random.choice(PICKUP_ONLY_TEMPLATES)
        pickup = random.choice(locations)

        text, entities = render_template(
            template,
            random.choice(REQUEST_PHRASES),
            pickup=pickup,
        )

        records.append(
            record(
                text,
                ["request_taxi"],
                entities,
            )
        )

    return records


def generate_dest_only(locations):
    records = []

    while len(records) < MAX_DEST_ONLY:
        template = random.choice(DEST_ONLY_TEMPLATES)
        dest = random.choice(locations)

        text, entities = render_template(
            template,
            random.choice(REQUEST_PHRASES),
            dest=dest,
        )

        records.append(
            record(
                text,
                ["request_taxi"],
                entities,
            )
        )

    return records



# ------------------------------------------------------------
# Standalone components
# ------------------------------------------------------------

def generate_standalone(values, intent, maximum):
    records = []

    for value in values[:maximum]:
        tokens = value.split()

        records.append({
            "text": value,
            "intents": [intent],
            "tokens": tokens,
            "ner_tags": ["O"] * len(tokens),
        })

    return records


# ------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------

def deduplicate(records):
    seen = set()
    result = []

    for r in records:
        key = (
            r["text"],
            tuple(r["intents"]),
            tuple(r["tokens"]),
            tuple(r["ner_tags"]),
        )

        if key not in seen:
            seen.add(key)
            result.append(r)

    return result


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_record(r):
    if len(r["tokens"]) != len(r["ner_tags"]):
        raise ValueError(
            f"Token/tag mismatch:\n{r}"
        )

    allowed = {
        "O",
        "B-LOCATION",
        "I-LOCATION",
    }

    invalid = set(r["ner_tags"]) - allowed

    if invalid:
        raise ValueError(
            f"Invalid NER tags {invalid}:\n{r}"
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    greetings, please, thanks, locations = load_components()

    print("Loaded language components:")
    print(f"  greetings: {len(greetings)}")
    print(f"  please:    {len(please)}")
    print(f"  thanks:    {len(thanks)}")
    print(f"  locations: {len(locations)}")

    records = []

    print("\nGenerating complete requests...")
    records.extend(
        generate_complete(
            locations,
            greetings,
            please,
            thanks,
        )
    )

    print("Generating pickup-only requests...")
    records.extend(
        generate_pickup_only(locations)
    )

    print("Generating destination-only requests...")
    records.extend(
        generate_dest_only(locations)
    )

    print("Generating standalone components...")
    records.extend(
        generate_standalone(
            greetings,
            "greeting",
            MAX_STANDALONE_GREETING,
        )
    )

    records.extend(
        generate_standalone(
            please,
            "politeness",
            MAX_STANDALONE_PLEASE,
        )
    )

    records.extend(
        generate_standalone(
            thanks,
            "thanks",
            MAX_STANDALONE_THANKS,
        )
    )

    print("\nValidating...")
    for r in records:
        validate_record(r)

    before = len(records)

    records = deduplicate(records)

    random.shuffle(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                ) + "\n"
            )

    print(f"Generated: {before}")
    print(f"After deduplication: {len(records)}")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
