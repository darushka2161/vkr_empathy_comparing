# src/emotion_classifier.py
# Singleton-обёртка над DistilBERT для классификации эмоций.
# Модель: bdotloh/distilbert-base-uncased-empathetic-dialogues-context
# 32 класса EmpatheticDialogues, ~250 MB, ~5-10ms на CPU.

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier

    from transformers import pipeline
    print("Loading DistilBERT emotion classifier "
          "(bdotloh/distilbert-base-uncased-empathetic-dialogues-context)...")
    _classifier = pipeline(
        "text-classification",
        model="bdotloh/distilbert-base-uncased-empathetic-dialogues-context",
        top_k=3,
    )
    print("Emotion classifier loaded.")
    return _classifier


def _extract_last_speaker_utterance(dialogue_context: str) -> str:
    """Извлекает последнюю реплику Speaker из контекста диалога."""
    lines = dialogue_context.strip().split("\n")
    for line in reversed(lines):
        if line.strip().lower().startswith("speaker:"):
            return line.split(":", 1)[1].strip()
    # Fallback: последняя непустая строка
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return dialogue_context


def classify_emotion(dialogue_context: str) -> dict:
    """
    Классифицирует эмоцию Speaker по последней его реплике в диалоге.

    Returns:
        {
            "emotion": str,           # top-1 emotion label
            "confidence": float,      # top-1 score
            "top3": [(str, float)],   # top-3 (label, score)
            "input_text": str,        # текст на вход классификатора
        }
    """
    clf = get_classifier()
    text = _extract_last_speaker_utterance(dialogue_context)

    raw = clf(text)  # list of dicts [{"label": ..., "score": ...}, ...]
    # pipeline с top_k и строковым вводом возвращает list of dicts
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    top3 = [(r["label"].lower().strip(), float(r["score"])) for r in raw]

    return {
        "emotion": top3[0][0],
        "confidence": top3[0][1],
        "top3": top3,
        "input_text": text,
    }
