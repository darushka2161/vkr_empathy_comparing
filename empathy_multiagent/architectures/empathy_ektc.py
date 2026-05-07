# architectures/empathy_ektc.py
# Архитектура: EmpathyEKTC
# Paper: "TOOL-ED: Enhancing Empathetic Response Generation with
#         the Tool Calling Capability of LLM" (COLING 2025)
#
# Pipeline: [Annotator LLM] → [COMET seq2seq] → [Reflector LLM] → [Generator LLM]
# LLM-вызовов: 2 (use_tool=False) или 3 (use_tool=True)
# COMET:  comet-atomic-2020-bart  — отдельная Seq2Seq-модель, не LLM

import asyncio
import nltk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Конфигурация COMET ────────────────────────────────────────────────────────
# HuggingFace model ID для COMET-ATOMIC-2020-BART.
# Альтернатива: укажи локальный путь к папке с весами.
COMET_MODEL_PATH = "mismayil/comet-atomic-2020-bart"

_RELATIONS  = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
_REL_LABELS = ["x_intent", "x_need",  "x_want",  "x_effect", "x_react"]

# Нормализация сокращений перед подачей в COMET (из оригинального app_comet.py)
_CONTRACTIONS = {
    "it's": "it is", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "you'd": "you would", "you're": "you are",
    "you'll": "you will", "i'm": "i am", "they're": "they are",
    "that's": "that is", "what's": "what is", "couldn't": "could not",
    "i've": "i have", "we've": "we have", "can't": "cannot",
    "i'd": "i would", "aren't": "are not", "isn't": "is not",
    "wasn't": "was not", "weren't": "were not", "won't": "will not",
    "there's": "there is", "there're": "there are",
}

_comet_instance = None  # lazy singleton


def _preprocess(text: str) -> str:
    """Lowercase + expand contractions + word-tokenize (как в оригинале)."""
    text = text.lower()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return " ".join(nltk.word_tokenize(text))


class _CometModel:
    """COMET-ATOMIC-2020-BART: seq2seq-модель для commonsense-инференса."""

    def __init__(self, model_path: str):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        print(f"[COMET] Loading {model_path} on {device_str}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        )
        self.model.eval()
        print("[COMET] Ready.")

    def _query(self, input_event: str, relation: str) -> list:
        """Запрос одного отношения. Возвращает 5 beam-гипотез."""
        query = f"{input_event} {relation} [GEN]"
        enc = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding="max_length"
        ).to(self.device)
        # убираем padding-столбцы (как в оригинальном comet.py)
        keep = enc["input_ids"].ne(self.tokenizer.pad_token_id).any(dim=0)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"][:, keep],
                attention_mask=enc["attention_mask"][:, keep],
                num_beams=5,
                num_return_sequences=5,
            )
        preds = self.tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # нормализуем выходы так же как оригинал (lower + word_tokenize + join)
        return [
            " ".join(nltk.word_tokenize(p.lower()))
            for p in preds if p.strip()
        ]

    def get_knowledge(self, dialogue_context: str) -> dict:
        """Возвращает {xIntent: [...], xNeed: [...], ...} — 5 значений на отношение."""
        event = _preprocess(dialogue_context)
        return {rel: self._query(event, rel) for rel in _RELATIONS}


def _get_comet() -> _CometModel:
    global _comet_instance
    if _comet_instance is None:
        _comet_instance = _CometModel(COMET_MODEL_PATH)
    return _comet_instance


# ── Промпты (из оригинальных constants.py и evaluation_knowledge.py) ──────────

_SYSTEM_PROMPT = (
    "This is an empathetic dialogue task: The first worker (Speaker) is given an "
    "emotion label and writes his own description of a situation when he has felt "
    "that way. Then, Speaker tells his story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "Listener should recognize and acknowledge others' feelings in a conversation as "
    "much as possible. You only need to provide the next round of response of Listener."
)

# Описания отношений из оригинального evaluation_knowledge.py
_RELATION_DESCRIPTIONS = (
    "xIntent represents their intent before the event.\n"
    "xNeed represents what they need in order for the event to happen.\n"
    "xWant represents what they would want after the event.\n"
    "xEffect represents the effect of the event on the person.\n"
    "xReact represents their reaction to the event."
)

# Преамбула перед знанием из оригинального evaluation_knowledge.py
_KNOWLEDGE_ORIGIN = (
    "Don't rush to reply, I can provide the following additional knowledge to help "
    "you provide a better reply. The following are the definitions of the five "
    "commonsense relations, followed by the content of the five relations extracted "
    "from the existing conversation. You can combine them and the dialogue context "
    "generates the final reply."
)

ANNOTATOR_SYSTEM = (
    "There are two roles in the conversation: Speaker and Listener.\n"
    "Assuming you are the Listener and you have access to an EmotionKnowledgeBase "
    "tool that provides commonsense knowledge about emotional situations.\n\n"
    "The tool returns the following five commonsense relations:\n"
    "  xIntent — the Speaker's likely intent before the event\n"
    "  xNeed   — what the Speaker needed for this event to happen\n"
    "  xWant   — what the Speaker wants after the event\n"
    "  xEffect — the effect of the event on the Speaker\n"
    "  xReact  — the Speaker's emotional reaction to the event\n\n"
    "Guidelines:\n"
    "1. Judge the emotional intensity of the Speaker based on the dialogue\n"
    "2. Decide whether using EmotionKnowledgeBase would help you give a more "
    "empathetic response\n\n"
    "Respond ONLY with JSON:\n"
    '{{"emotional_intensity": "low/medium/high", "use_tool": true/false, '
    '"reason": "one sentence explanation"}}'
)

REFLECTOR_SYSTEM = (
    "There are two roles in the conversation: Speaker and Listener.\n"
    "The Listener used the EmotionKnowledgeBase tool and received the following "
    "commonsense knowledge:\n\n{knowledge}\n\n"
    "Check relevance via three consistency types:\n"
    "1. Causal consistency    — does the knowledge align with the cause of the "
    "Speaker's feelings?\n"
    "2. Intent consistency    — does the knowledge reflect the Speaker's "
    "intentions and needs?\n"
    "3. Emotional consistency — does the knowledge match the emotional tone?\n\n"
    "Respond ONLY with JSON:\n"
    '{{"causal_consistent": true/false, "intent_consistent": true/false, '
    '"emotional_consistent": true/false, "use_knowledge": true/false, '
    '"reason": "one sentence explanation"}}'
)

GENERATOR_NO_KNOWLEDGE = (
    _SYSTEM_PROMPT + "\n\n"
    "Provide your next empathetic response. "
    "1-3 sentences, natural and conversational, no emoji.\n"
    "Respond with ONLY the Listener's response text."
)

# Шаблон Generator со знанием воспроизводит формат из evaluation_knowledge.py:
# content + knowledge_origin + descriptions + relation_content
GENERATOR_WITH_KNOWLEDGE = (
    _SYSTEM_PROMPT + "\n\n"
    "{knowledge_block}\n\n"
    "Provide your next empathetic response using this knowledge. "
    "1-3 sentences, natural and conversational, no emoji.\n"
    "Respond with ONLY the Listener's response text."
)


def _build_knowledge_block(knowledge: dict) -> str:
    """Форматирует знание в точности как в оригинале (app_comet.py, evaluation_knowledge.py)."""
    rel_lines = ""
    for rel, label in zip(_RELATIONS, _REL_LABELS):
        rel_lines += f"{label}: {knowledge.get(rel, [])}\n"
    return f"{_KNOWLEDGE_ORIGIN}\n{_RELATION_DESCRIPTIONS}\n{rel_lines}"


async def empathy_ektc(dialogue_context: str, llm) -> dict:
    llm_calls = 0

    # ── Шаг 1: Annotator — нужно ли звать COMET? ─────────────────────────────
    annotator_data = await llm.generate_json(
        ANNOTATOR_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=128,
    )
    llm_calls += 1

    use_tool = annotator_data.get("use_tool", False)
    knowledge_data = {}
    reflector_data = {}
    use_knowledge = False

    if use_tool:
        # ── Шаг 2: COMET — commonsense-инференс (не LLM!) ────────────────────
        # run_in_executor чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        comet = _get_comet()
        knowledge_data = await loop.run_in_executor(
            None, comet.get_knowledge, dialogue_context
        )

        # ── Шаг 3: Reflector — проверяем консистентность ─────────────────────
        knowledge_str = _build_knowledge_block(knowledge_data)
        reflector_data = await llm.generate_json(
            REFLECTOR_SYSTEM.format(knowledge=knowledge_str),
            f"Dialogue:\n{dialogue_context}",
            max_tokens=128,
        )
        llm_calls += 1
        use_knowledge = reflector_data.get("use_knowledge", False)

    # ── Шаг 4: Generator ─────────────────────────────────────────────────────
    if use_knowledge:
        system = GENERATOR_WITH_KNOWLEDGE.format(
            knowledge_block=_build_knowledge_block(knowledge_data)
        )
    else:
        system = GENERATOR_NO_KNOWLEDGE

    response = await llm.generate(
        system,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=256,
        temperature=0.1,
    )
    llm_calls += 1

    return {
        "response": response,
        "annotator": annotator_data,
        "knowledge": knowledge_data,
        "reflector": reflector_data,
        "used_knowledge": use_knowledge,
        "llm_calls": llm_calls,
    }
