# architectures/empathy_trace.py
# Архитектура TRACE (Liu et al., 2025 — arXiv:2509.21849)
#
# Pipeline (4 LLM-вызова + RAG):
#   Dialogue
#     → [1. ASI — Affective State Identifier]     LLM-вызов 1  (Ekman emotion)
#     → [2. CAE — Causal Analysis Engine]          LLM-вызов 2  (trigger + cause)
#     → [3. SRP — Strategic Response Planner] + RAG  LLM-вызов 3  (ER / IP / EX)
#     → [4. ERS — Empathetic Response Synthesizer] + RAG  LLM-вызов 4
#     → Response

import numpy as np
from architectures.empathy_rag import EmpathyRetriever

# ── Маппинг 32 эмоций ED → 6 категорий Экмана ───────────────────────────────
# Источник: оригинальный Agent1.py из репозитория TRACE
EKMAN_MAP: dict[str, str] = {
    "joyful":        "happiness", "excited":      "happiness", "proud":        "happiness",
    "grateful":      "happiness", "content":      "happiness", "impressed":    "happiness",
    "caring":        "happiness", "trusting":     "happiness", "faithful":     "happiness",
    "hopeful":       "happiness", "confident":    "happiness", "sentimental":  "happiness",
    "sad":           "sadness",   "lonely":       "sadness",   "disappointed": "sadness",
    "devastated":    "sadness",   "nostalgic":    "sadness",   "guilty":       "sadness",
    "ashamed":       "sadness",   "embarrassed":  "sadness",
    "angry":         "anger",     "furious":      "anger",     "annoyed":      "anger",
    "jealous":       "anger",
    "afraid":        "fear",      "terrified":    "fear",      "anxious":      "fear",
    "apprehensive":  "fear",
    "disgusted":     "disgust",
    "surprised":     "surprise",  "anticipating": "surprise",  "prepared":     "surprise",
}

# Обратный маппинг: Ekman-категория → список fine-grained эмоций ED
EKMAN_TO_FINE: dict[str, list[str]] = {}
for fine, ekman in EKMAN_MAP.items():
    EKMAN_TO_FINE.setdefault(ekman, []).append(fine)

# ── Промпт Агент 1: ASI ───────────────────────────────────────────────────────
_EMOTION_DEFINITIONS = "\n".join(
    f"- **{ekman.capitalize()}**: can represent finer emotions such as: {', '.join(fines)}."
    for ekman, fines in EKMAN_TO_FINE.items()
)

ASI_SYSTEM = (
    "You are a psychology expert specializing in emotion classification. "
    "Your task is to analyze a dialogue and identify the subject's core emotion.\n\n"
    "IMPORTANT: The emotion labels were originally 32 fine-grained emotions mapped to six standard Ekman categories. "
    "You must understand the nuances within each broad category.\n\n"
    f"Definitions of the six categories:\n{_EMOTION_DEFINITIONS}\n\n"
    "Analyze the conversation's context, flow, and word choice. "
    "For example, a conversation expressing 'sentimental' should be classified as 'happiness', "
    "and one expressing 'annoyed' should be classified as 'anger'.\n\n"
    "Respond ONLY with JSON: "
    '{"reasoning": "brief explanation", '
    '"final_emotion": "happiness or sadness or anger or fear or disgust or surprise"}'
)

# ── Промпт Агент 2: CAE ───────────────────────────────────────────────────────
_CAUSE_TAXONOMY = [
    "Social Connection & Affection",
    "Conflict & Injustice",
    "Loss & Failure",
    "Achievement & Success",
    "Threat & Potential Danger",
    "Unexpectedness & Novelty",
]

CAE_SYSTEM = (
    "You are a highly perceptive psychological analyst.\n\n"
    "BACKGROUND CONTEXT: The dialogue comes from a controlled experiment where a subject (the 'Seeker') "
    "is in a specific emotional state while talking to an experimenter (the 'Responder'). "
    "A preliminary expert agent has already identified the dominant Ekman emotion. "
    "The emotion label you receive is the result of that initial nuanced analysis.\n\n"
    "YOUR TASK — two sub-tasks in a single JSON response:\n"
    "1. Identify Trigger Spans: extract 1–2 most crucial sentences spoken by the Seeker "
    "that directly trigger or most strongly express the given emotion.\n"
    "2. Summarize and Categorize the Global Cause:\n"
    "   a. Write a one-sentence summary explaining the overall reason for the emotion.\n"
    f"   b. Classify into ONE of these categories: {', '.join(_CAUSE_TAXONOMY)}.\n\n"
    "Respond ONLY with JSON:\n"
    '{{"trigger_spans": ["sentence1"], '
    '"global_cause_summary": "one-sentence summary", '
    f'"cause_category": "one of {_CAUSE_TAXONOMY}"}}'
)

# ── Промпт Агент 3: SRP ───────────────────────────────────────────────────────
_STRATEGY_DEFINITIONS = (
    "- ER (Emotional Reaction): Expressing your own emotions (warmth, compassion, concern) "
    "experienced after reading the seeker's situation. Establishes empathic rapport. "
    'A strong response explicitly labels the felt emotion (e.g., "I feel really sad for you").\n'
    "- IP (Interpretation): Communicating a cognitive understanding of the feelings and experiences "
    "inferred from the seeker's situation. Can specify the inferred feeling "
    '(e.g., "This must be terrifying") or describe a similar personal experience.\n'
    "- EX (Exploration): Improving understanding of the seeker by exploring feelings and experiences "
    "not explicitly stated. Shows active interest by asking a specific, gentle probing question "
    '(e.g., "Are you feeling alone right now?").'
)

SRP_SYSTEM = (
    "You are an expert psychological counselor, acting as the final decision-making agent "
    "in a three-stage analysis pipeline. Your role is to synthesize previous findings "
    "and decide on the most appropriate empathetic strategy.\n\n"
    "ANALYSIS PIPELINE OVERVIEW:\n"
    "1. Agent 1 (Emotion Classifier): identified the subject's dominant Ekman emotion.\n"
    "2. Agent 2 (Cause Analyst): identified trigger spans, global cause summary, and cause category.\n\n"
    "HISTORICAL CASE REFERENCE:\n"
    "Below are relevant historical success cases for reference. "
    "Analyze the strategies used for inspiration, but your decision must be tailored "
    "to the current, unique conversation.\n"
    "{rag_examples}\n\n"
    "YOUR TASK: choose the SINGLE most appropriate empathetic response strategy. "
    "Do NOT write the actual response.\n\n"
    f"Response Strategies:\n{_STRATEGY_DEFINITIONS}\n\n"
    'Respond ONLY with JSON: {{"reasoning_for_choice": "brief expert explanation", '
    '"chosen_strategy": "ER or IP or EX"}}'
)

# ── Промпт Агент 4: ERS ───────────────────────────────────────────────────────
ERS_SYSTEM = (
    "You are an expert and compassionate AI counselor. "
    "Your role is to synthesize a complete case file and generate a final nuanced empathetic response.\n\n"
    "ANALYSIS PIPELINE SUMMARY:\n"
    "1. Agent 1 (Emotion Classifier): identified the subject's dominant emotion.\n"
    "2. Agent 2 (Cause Analyst): pinpointed the emotional trigger and root cause.\n"
    "3. Agent 3 (Strategy Decider): recommended a primary response strategy.\n\n"
    "RESPONSE GENERATION GUIDELINES:\n"
    "Reference the provided successful examples below. "
    "Emulate their tone, phrasing, and conciseness while keeping your response original.\n\n"
    "--- BEGIN EXAMPLES ---\n"
    "{rag_examples}\n"
    "--- END EXAMPLES ---\n\n"
    "STRATEGY BLENDING: Merely using the primary strategy is not enough. "
    "Start with the Primary Strategy, then naturally integrate a Secondary Strategy.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "- Context-Awareness (THE GOLDEN RULE): Review the Emotion Trigger Sentences, "
    "select a specific concrete detail (e.g., 'my puppy', 'my boss'), "
    "and weave it into your response to prove you are listening. This is non-negotiable.\n"
    "- Strategy-Guided: primary strategy as main theme, skillfully blended with a secondary one.\n"
    "- Empathetic and Natural: supportive, human-like tone.\n"
    "- Non-judgmental: do NOT give unsolicited advice.\n"
    "- CRITICAL LENGTH RULE: 1-2 sentences, maximum 15 words. Do NOT write long paragraphs.\n\n"
    'Respond ONLY with JSON: {{"final_empathetic_response": "response text"}}'
)


# ── RAG helpers ───────────────────────────────────────────────────────────────

def _retrieve_for_trace(retriever: EmpathyRetriever, dialogue_context: str,
                        ekman_emotion: str, top_k: int = 2) -> list:
    """
    Эмулирует оригинальный TRACE: ChromaDB фильтрует сразу весь Ekman-кластер
    ({"emotion_id": ekman_emotion}), затем ранжирует по сходству.

    Здесь индексы разбиты по fine-grained ED-эмоциям, поэтому:
    - собираем кандидатов из ВСЕХ fine-grained индексов нужной Ekman-группы,
    - объединяем в один пул и берём глобальный top-k по similarity.

    Отличие от оригинала: embedding-модель all-MiniLM-L6-v2 вместо BAAI/bge-large-en-v1.5.
    Для перехода на bge-large нужно перестроить индекс через build_index.py.
    """
    fine_emotions = EKMAN_TO_FINE.get(ekman_emotion, [])

    # Собираем кандидатов из ВСЕХ fine-grained индексов этой Ekman-группы
    all_candidates = []
    for fine_em in fine_emotions:
        if fine_em not in retriever.emotion_indices:
            continue
        # top_k*2 с запасом — после объединения останется лучший top_k
        results = retriever.retrieve(dialogue_context, emotion=fine_em, top_k=top_k * 2)
        all_candidates.extend(results)

    if all_candidates:
        # Глобальное ранжирование по similarity → top-k (как в оригинале)
        all_candidates.sort(key=lambda x: -x.get("similarity", 0))
        return all_candidates[:top_k]

    # Fallback: семантический поиск без фильтра по эмоции
    return retriever._fallback_retrieve(dialogue_context, top_k=top_k)


def _format_rag_examples(cases: list) -> str:
    if not cases:
        return "No relevant examples found."
    parts = ["### RELEVANT CASE EXAMPLES FROM KNOWLEDGE BASE"]
    for i, case in enumerate(cases, 1):
        dialogue = case.get("context", "N/A")
        emotion = case.get("emotion", "N/A")
        gold = case.get("gold_response", "N/A")
        parts.append(
            f"\n--- Example {i} ---\n"
            f"SITUATION (Emotion: {emotion}):\n{dialogue}\n"
            f"SUCCESSFUL EMPATHETIC RESPONSE:\n{gold}\n"
            "--- End of Example ---"
        )
    return "\n".join(parts)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def empathy_trace(dialogue_context: str, llm, retriever: EmpathyRetriever) -> dict:
    # ── Agent 1: ASI — Affective State Identifier (LLM-вызов 1) ──────────────
    emotion_data = await llm.generate_json(
        ASI_SYSTEM,
        f"Please classify the following conversation:\n\n\"{dialogue_context}\"",
        temperature=0.0,
        max_tokens=200,
    )
    if isinstance(emotion_data, list):
        emotion_data = emotion_data[0] if emotion_data else {}
    ekman_emotion = emotion_data.get("final_emotion", "sadness").lower().strip()
    # Нормализация — если вернул что-то неожиданное
    if ekman_emotion not in EKMAN_TO_FINE:
        ekman_emotion = "sadness"

    # ── Agent 2: CAE — Causal Analysis Engine (LLM-вызов 2) ──────────────────
    cause_data = await llm.generate_json(
        CAE_SYSTEM,
        f"Dialogue:\n{dialogue_context}\n\nIdentified Emotion: {ekman_emotion}\n\nYour Analysis (JSON):",
        temperature=0.0,
        max_tokens=300,
    )
    if isinstance(cause_data, list):
        cause_data = cause_data[0] if cause_data else {}

    trigger_spans = cause_data.get("trigger_spans", [])
    cause_summary = cause_data.get("global_cause_summary", "")
    cause_category = cause_data.get("cause_category", "")

    # ── RAG retrieval (используется агентами 3 и 4) ───────────────────────────
    retrieved = _retrieve_for_trace(retriever, dialogue_context, ekman_emotion, top_k=2)
    rag_examples_str = _format_rag_examples(retrieved)

    # ── Agent 3: SRP — Strategic Response Planner (LLM-вызов 3) ─────────────
    strategy_data = await llm.generate_json(
        SRP_SYSTEM.format(rag_examples=rag_examples_str),
        (
            f"CASE FILE:\n\n"
            f"1. Original Dialogue:\n{dialogue_context}\n\n"
            f"2. Agent 1 Analysis — Identified Emotion: {ekman_emotion}\n\n"
            f"3. Agent 2 Analysis:\n"
            f"   - Trigger Sentences: {trigger_spans}\n"
            f"   - Cause Summary: {cause_summary}\n"
            f"   - Cause Category: {cause_category}\n\n"
            "Your Analysis (JSON):"
        ),
        temperature=0.0,
        max_tokens=200,
    )
    if isinstance(strategy_data, list):
        strategy_data = strategy_data[0] if strategy_data else {}
    chosen_strategy = strategy_data.get("chosen_strategy", "ER").upper().strip()
    if chosen_strategy not in ("ER", "IP", "EX"):
        chosen_strategy = "ER"

    # ── Agent 4: ERS — Empathetic Response Synthesizer (LLM-вызов 4) ─────────
    response_data = await llm.generate_json(
        ERS_SYSTEM.format(rag_examples=rag_examples_str),
        (
            f"CASE FILE:\n\n"
            f"1. Original Dialogue:\n{dialogue_context}\n\n"
            f"2. Agent 1 Analysis — Identified Emotion: {ekman_emotion}\n\n"
            f"3. Agent 2 Analysis:\n"
            f"   - Trigger Sentences: {trigger_spans}\n"
            f"   - Cause Summary: {cause_summary}\n"
            f"   - Cause Category: {cause_category}\n\n"
            f"4. Agent 3 Analysis — Chosen Primary Strategy: {chosen_strategy}\n\n"
            "Your Response (JSON):"
        ),
        temperature=0.5,
        max_tokens=256,
    )
    if isinstance(response_data, list):
        response_data = response_data[0] if response_data else {}

    response = response_data.get("final_empathetic_response", "")

    return {
        "response": response,
        "ekman_emotion": ekman_emotion,
        "emotion_reasoning": emotion_data.get("reasoning", ""),
        "cause": cause_data,
        "strategy": strategy_data,
        "retrieved_examples": [
            {
                "context_preview": r["context"][:120] if r.get("context") else "",
                "gold_response": r.get("gold_response", ""),
                "emotion": r.get("emotion", ""),
                "similarity": r.get("similarity", 0),
            }
            for r in retrieved
        ],
        "llm_calls": 4,
    }
