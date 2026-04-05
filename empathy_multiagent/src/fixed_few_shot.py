# src/fixed_few_shot.py
# 5 фиксированных few-shot примеров из ED train для использования
# в финальных генераторах empathy_chain, empathy_debate, empathy_loop.
#
# Состав: 1 позитивная эмоция, 1 нейтральная, 3 негативных.
# Выбираются один раз при первом вызове get_few_shot_block(), seed=42.

import random

# Валентность эмоций из EmpatheticDialogues
_POSITIVE = {
    "excited", "proud", "grateful", "hopeful", "confident", "joyful",
    "trusting", "faithful", "prepared", "anticipating", "content",
    "impressed", "surprised",
}
_NEUTRAL = {
    "nostalgic", "sentimental", "caring",
}
_NEGATIVE = {
    "afraid", "anxious", "apprehensive", "embarrassed", "ashamed",
    "devastated", "sad", "disappointed", "lonely", "jealous",
    "disgusted", "angry", "furious", "annoyed", "terrified", "guilty",
}


def _pick_examples(train_examples: list, seed: int = 42) -> list:
    """
    Выбирает 5 примеров: 1 positive, 1 neutral, 3 negative.
    Возвращает список dict с полями: emotion, valence, context, gold_response.
    """
    rng = random.Random(seed)

    by_valence: dict = {"positive": [], "neutral": [], "negative": []}
    for ex in train_examples:
        em = ex["emotion"].lower().strip()
        if em in _POSITIVE:
            by_valence["positive"].append(ex)
        elif em in _NEUTRAL:
            by_valence["neutral"].append(ex)
        elif em in _NEGATIVE:
            by_valence["negative"].append(ex)

    selected = []
    for valence, count in [("positive", 1), ("neutral", 1), ("negative", 3)]:
        pool = by_valence[valence]
        if not pool:
            continue
        for ex in rng.sample(pool, min(count, len(pool))):
            selected.append({
                "emotion": ex["emotion"],
                "valence": valence,
                "context": ex["context"],
                "gold_response": ex["gold_response"],
            })

    return selected


def _format_examples(examples: list) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        # Берём последние 3 реплики контекста чтобы не раздувать промпт
        lines = ex["context"].strip().split("\n")
        short_ctx = "\n".join(lines[-3:]) if len(lines) > 3 else ex["context"]
        parts.append(
            f"Example {i} ({ex['valence']} — {ex['emotion']}):\n"
            f"Dialogue:\n{short_ctx}\n"
            f"Listener: {ex['gold_response']}"
        )
    return "\n\n".join(parts)


# Кэш: инициализируется один раз
_few_shot_block: str | None = None
_raw_examples: list | None = None


def get_few_shot_block() -> str:
    """
    Возвращает отформатированный блок few-shot примеров.
    Загружает train-сет при первом вызове.
    """
    global _few_shot_block, _raw_examples
    if _few_shot_block is not None:
        return _few_shot_block

    from src.load_dataset import prepare_examples
    print("Loading train examples for fixed few-shot block...")
    train = prepare_examples("train")
    _raw_examples = _pick_examples(train, seed=42)
    _few_shot_block = (
        "EXAMPLES of real empathetic Listener responses "
        "(positive, neutral, and negative emotion situations):\n\n"
        + _format_examples(_raw_examples)
        + "\n\n---\nNow respond to the NEW dialogue below."
    )
    emotions = [f"{ex['emotion']} ({ex['valence']})" for ex in _raw_examples]
    print(f"Fixed few-shot block ready: {emotions}")
    return _few_shot_block


def get_raw_examples() -> list:
    """Возвращает список выбранных примеров (для логирования)."""
    if _raw_examples is None:
        get_few_shot_block()
    return _raw_examples
