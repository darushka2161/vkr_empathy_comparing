# architectures/empathy_few_shot.py
# Архитектура 5: FewShot
# Dialogue → [Listener + K примеров из train] → Response
# LLM-вызовов: 1
# Примеры берутся случайно из train-сета, сгруппированы по эмоции

import random

SYSTEM_PROMPT = (
    "This is an empathetic dialogue task: The first worker (Speaker) is given an "
    "emotion label and writes his own description of a situation when he has felt "
    "that way. Then, Speaker tells his story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "Listener should recognize and acknowledge others' feelings in a conversation as "
    "much as possible. You are an empathetic conversational AI chatbot that can "
    "empathize with users. You only need to provide the next round of response of "
    "Listener. Reply with 1-3 sentences, natural and conversational, no emoji."
)

FEW_SHOT_TEMPLATE = """{system}

Below are {k} examples of empathetic dialogues and good listener responses:

{examples}

Now respond to the following dialogue. Respond with ONLY the listener's reply text."""


def _format_examples(examples: list) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        # Берём последние 3 реплики контекста чтобы не раздувать промпт
        lines = ex["context"].strip().split("\n")
        short_ctx = "\n".join(lines[-3:]) if len(lines) > 3 else ex["context"]
        parts.append(
            f"Example {i}:\n"
            f"Dialogue: {short_ctx}\n"
            f"Listener: {ex['gold_response']}"
        )
    return "\n\n".join(parts)


class FewShotSampler:
    """Хранит train-примеры сгруппированными по эмоции для быстрой выборки."""

    def __init__(self, train_examples: list, k: int = 3, seed: int = 42):
        self.k = k
        self.rng = random.Random(seed)
        self.by_emotion: dict = {}
        self.all_examples = train_examples
        for ex in train_examples:
            em = ex["emotion"].lower().strip()
            self.by_emotion.setdefault(em, []).append(ex)

    def sample(self, emotion: str = None) -> list:
        """Возвращает k примеров: сначала по эмоции, иначе случайно."""
        emotion = (emotion or "").lower().strip()
        pool = self.by_emotion.get(emotion, self.all_examples)
        k = min(self.k, len(pool))
        return self.rng.sample(pool, k)


# Глобальный сэмплер — инициализируется один раз при первом вызове
_sampler: FewShotSampler | None = None


def get_sampler(k: int = 3) -> FewShotSampler:
    global _sampler
    if _sampler is None:
        from src.load_dataset import prepare_examples
        print("Loading train examples for few-shot sampler...")
        train = prepare_examples("train")
        _sampler = FewShotSampler(train, k=k)
        print(f"Few-shot sampler ready: {len(train)} train examples")
    return _sampler


async def empathy_few_shot(dialogue_context: str, llm, k: int = 3) -> dict:
    sampler = get_sampler(k=k)

    # Нет эмоции на этапе выборки — берём случайные примеры
    examples = sampler.sample(emotion=None)

    system = FEW_SHOT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        k=k,
        examples=_format_examples(examples),
    )

    response = await llm.generate(
        system,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=256,
        temperature=0.1,
    )

    return {
        "response": response,
        "few_shot_examples": [
            {"context_preview": ex["context"][-100:], "gold_response": ex["gold_response"]}
            for ex in examples
        ],
        "llm_calls": 1,
    }
