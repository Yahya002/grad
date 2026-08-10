# Arabic NLP Dataset Specification

## 1. Purpose

This dataset is used to train two independent NLP models:

1. A multi-label intent classifier.
2. A named-entity recognition (NER) model.

Both models are trained from the same examples.

---

## 2. Canonical Format

The master dataset is JSON Lines (`.jsonl`).

Each line represents exactly one user utterance.

Example:

{
  "text": "بدي تكسي من ساحة البريد لـ المشفى الكندي بكرا الصبح",
  "intents": [
    "taxi"
  ],
  "tokens": [
    "بدي",
    "تكسي",
    "من",
    "ساحة",
    "البريد",
    "لـ",
    "المشفى",
    "الكندي",
    "بكرا",
    "الصبح"
  ],
  "ner_tags": [
    "O",
    "O",
    "O",
    "B-PICKUP",
    "I-PICKUP",
    "O",
    "B-DEST",
    "I-DEST",
    "B-TIME",
    "I-TIME"
  ]
}

---

## 3. Fields

### `text`

The original user utterance.

Rules:

- Preserve the original wording.
- Preserve Syrian Arabic dialect.
- Do not translate.
- Do not grammatically "correct" the user.
- Do not remove colloquial expressions.
- Do not normalize the text unless the dataset-maintainer explicitly
  decides to introduce normalization.

---

### `intents`

An array containing zero or more intent labels.

Examples:

Single intent:

"intents": ["taxi"]

Multiple intents:

"intents": ["taxi", "urgent"]

No recognized intent:

"intents": []

Rules:

- Multiple intents are allowed.
- Intent order has no semantic meaning.
- Do not duplicate an intent.
- Only use intent names from `config/intents.json`.
- Do not invent new intent names while annotating.
- If a new intent is needed, propose it to the dataset maintainer
  instead of adding it directly.

---

### `tokens`

The utterance split into annotation tokens.

Rules:

- Tokens must appear in the same order as in `text`.
- Do not remove words.
- Do not add words.
- Keep the tokenization consistent across the dataset.

---

### `ner_tags`

One BIO tag for every token.

The following invariant MUST always hold:

len(tokens) == len(ner_tags)

---

## 4. Intent Annotation

Intent describes what the user is trying to accomplish.

Intent is sentence-level.

Multiple intents are allowed.

Example:

"بدي تكسي بسرعة"

may have:

"intents": ["taxi", "urgent"]

Do not assume that only one intent can exist.

### No intent

If none of the defined intents applies:

"intents": []

Do NOT create an `other` intent unless it is explicitly added to
`config/intents.json`.

---

## 5. NER Annotation

NER uses BIO tagging.

### `O`

The token is not part of an entity.

### `B-X`

The token begins an entity of type X.

### `I-X`

The token continues an entity of type X.

---

## 6. Current Entity Types

The current entity vocabulary is:

- PICKUP
- DEST
- TIME

Therefore the valid NER tags are:

- O
- B-PICKUP
- I-PICKUP
- B-DEST
- I-DEST
- B-TIME
- I-TIME

---

## 7. PICKUP

PICKUP identifies the place from which the requested service begins.

Example:

"بدي تكسي من حي الرمل للمرفأ"

Tokens:

حي      B-PICKUP
الرمل   I-PICKUP

PICKUP is a semantic role.

It is NOT a permanent property of a location.

The same place can be PICKUP in one utterance and DEST in another.

---

## 8. DEST

DEST identifies the destination of the requested service.

Example:

"بدي تكسي من حي الرمل للمرفأ"

Tokens:

المرفأ   B-DEST

DEST is a semantic role, not a geographical entity category.

---

## 9. TIME

TIME identifies an expression referring to when the requested action
should happen.

Annotate the complete expression when multiple words collectively
express the time.

Example:

"بكرا الصبح"

becomes:

بكرا     B-TIME
الصبح    I-TIME

Example:

"بعد ربع ساعة تقريباً"

becomes:

بعد       B-TIME
ربع       I-TIME
ساعة      I-TIME
تقريباً   I-TIME

---

## 10. No NER Entity

Utterances without entities are valid.

Example:

{
  "text": "بدي حدا ياخدني",
  "intents": ["taxi"],
  "tokens": [
    "بدي",
    "حدا",
    "ياخدني"
  ],
  "ner_tags": [
    "O",
    "O",
    "O"
  ]
}

Do NOT remove examples simply because they contain no NER entities.

---

## 11. Entity Boundaries

Annotate the complete meaningful span.

For example:

"ساحة البريد"

should be:

ساحة     B-PICKUP
البريد   I-PICKUP

not:

ساحة     B-PICKUP
البريد   O

Similarly:

"المشفى الكندي"

should be:

المشفى   B-DEST
الكندي   I-DEST

---

## 12. Colloquial Language

Keep local Syrian Arabic wording.

Examples such as:

- بدي
- دبرولي
- جيبولي
- عالخمسة
- عالتمانية
- هالمسا
- عـ

are valid training material.

Do not replace them with Modern Standard Arabic.

---

## 13. Dataset Quality Rules

Every record MUST satisfy:

1. Valid JSON.
2. Required fields are present.
3. `text` is a string.
4. `intents` is an array.
5. Every intent exists in `config/intents.json`.
6. `tokens` is an array.
7. `ner_tags` is an array.
8. `len(tokens) == len(ner_tags)`.
9. Every NER tag is valid.
10. Intents contain no duplicates.
11. The original text is preserved.
12. No new labels are invented during annotation.

---

## 14. Adding New Labels

Contributors MUST NOT directly create new labels.

If a new intent or entity type appears necessary:

1. Record the example.
2. Propose the new label.
3. Have the dataset maintainer approve it.
4. Add it to the corresponding configuration file.
5. Update this specification.
6. Revalidate the dataset.

---

## 15. Dataset Splitting

Contributors work only on:

data/raw/master.jsonl

The training pipeline creates:

data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl

Contributors should not manually edit these generated files.

---

## 16. Source Priority

When procuring data, prefer:

1. Real user utterances.
2. Manually written local/domain-specific examples.
3. Controlled synthetic variations.

Synthetic examples should imitate realistic local language rather than
generic textbook Arabic.

---

## 17. Important Principle

The goal is not to create grammatically perfect Arabic.

The goal is to teach the models how actual users express requests in the
target domain.