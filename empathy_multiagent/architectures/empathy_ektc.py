# architectures/empathy_ektc.py
# Архитектура: EmpathyEKTC
# Inspired by: "TOOL-ED: Enhancing Empathetic Response Generation with
#               the Tool Calling Capability of LLM" (COLING 2025)
#
# Pipeline: [Annotator] → (if needed) [KnowledgeGen] → [Reflector] → [Generator]
# LLM-вызовов: 2 (low intensity) или 4 (high intensity + knowledge used)
#
# Ключевая идея из статьи: LLM сам решает, нужно ли внешнее знание,
# а Reflector фильтрует нерелевантное знание по трём критериям консистентности.
# Вместо COMET-модели знание генерирует сам LLM (те же 5 commonsense-отношений).

_TASK_CONTEXT = (
    "This is an empathetic dialogue task. The Speaker shares a personal situation "
    "and their feelings in a conversation with a Listener. The Listener's role is to "
    "recognize and acknowledge the Speaker's feelings as much as possible."
)

# ── Агент 1: Annotator ────────────────────────────────────────────────────────
# Оценивает эмоциональную интенсивность и решает нужно ли знание.
# Промпт основан на точном тексте из статьи (Section 3.1, Annotator).
ANNOTATOR_SYSTEM = (
    "There are two roles in the conversation: Speaker and Listener.\n"
    "Assuming you are the Listener and you have access to an EmotionKnowledgeBase tool "
    "that provides commonsense knowledge about emotional situations.\n\n"
    "The tool can give you the following information based on the dialogue context:\n"
    "  xIntent — the Speaker's likely intent before the event\n"
    "  xNeed   — what the Speaker needed for this event to happen\n"
    "  xWant   — what the Speaker wants after the event\n"
    "  xEffect — the effect of the event on the Speaker\n"
    "  xReact  — the Speaker's emotional reaction to the event\n\n"
    "Please follow these guidelines:\n"
    "1. Judge the emotional intensity of the Speaker based on the dialogue\n"
    "2. Based on the emotional intensity, decide whether using the EmotionKnowledgeBase "
    "would genuinely help you provide a more empathetic and informed response\n\n"
    "Respond ONLY with JSON:\n"
    '{{"emotional_intensity": "low/medium/high", "use_tool": true/false, '
    '"reason": "one sentence explanation"}}'
)

# ── Агент 2: KnowledgeGen (заменяет COMET из статьи) ─────────────────────────
# Генерирует те же 5 commonsense-отношений что COMET, но через LLM.
KNOWLEDGE_SYSTEM = (
    "You are an EmotionKnowledgeBase. Given a dialogue context, generate "
    "commonsense inferences about the Speaker's emotional situation.\n\n"
    "Generate 3-5 likely values for each relation:\n"
    "  xIntent — what was the Speaker's intent or goal before this event?\n"
    "  xNeed   — what did the Speaker need for this situation to happen?\n"
    "  xWant   — what does the Speaker want now, after this event?\n"
    "  xEffect — what effect does this event have on the Speaker?\n"
    "  xReact  — how does the Speaker emotionally react to this?\n\n"
    "Keep values short (2-5 words each), realistic, and grounded in the dialogue.\n\n"
    "Respond ONLY with JSON:\n"
    '{{"xIntent": ["...", "..."], "xNeed": ["...", "..."], "xWant": ["...", "..."], '
    '"xEffect": ["...", "..."], "xReact": ["...", "..."]}}'
)

# ── Агент 3: Reflector ────────────────────────────────────────────────────────
# Проверяет консистентность знания по 3 критериям (Section 3.1, Reflector).
REFLECTOR_SYSTEM = (
    "There are two roles in the conversation: Speaker and Listener.\n"
    "The Listener used an EmotionKnowledgeBase tool and received the following knowledge.\n\n"
    "Evaluate the relevance of this knowledge for generating an empathetic response "
    "by checking three types of consistency:\n"
    "1. Causal consistency   — does the knowledge align with the likely cause of "
    "the Speaker's feelings?\n"
    "2. Intent consistency   — does the knowledge correctly reflect the Speaker's "
    "intentions and needs?\n"
    "3. Emotional consistency — does the knowledge match the emotional tone and "
    "context of the dialogue?\n\n"
    "Knowledge generated:\n{knowledge}\n\n"
    "Please follow these guidelines:\n"
    "1. Assess each of the three consistency types\n"
    "2. Based on the consistency reflection, decide whether this knowledge would "
    "genuinely improve the empathetic response\n\n"
    "Respond ONLY with JSON:\n"
    '{{"causal_consistent": true/false, "intent_consistent": true/false, '
    '"emotional_consistent": true/false, "use_knowledge": true/false, '
    '"reason": "one sentence explanation"}}'
)

# ── Агент 4: Generator (без знания) ─────────────────────────────────────────
GENERATOR_NO_KNOWLEDGE = (
    _TASK_CONTEXT + "\n\n"
    "You are the Listener. Provide your next empathetic response.\n"
    "Rules: 1-3 sentences, natural and conversational, no emoji.\n"
    "Recognize and acknowledge the Speaker's feelings genuinely.\n"
    "You only need to provide the next round of response of Listener.\n\n"
    "Respond with ONLY the Listener's response text."
)

# ── Агент 4: Generator (со знанием) ─────────────────────────────────────────
# ReAct-стиль из статьи: Observation → Response
GENERATOR_WITH_KNOWLEDGE = (
    _TASK_CONTEXT + "\n\n"
    "You are the Listener. You have used the EmotionKnowledgeBase tool and received "
    "the following commonsense knowledge about the Speaker's situation:\n\n"
    "  Speaker's likely intent before this event : {xIntent}\n"
    "  What the Speaker needed                   : {xNeed}\n"
    "  What the Speaker wants now                : {xWant}\n"
    "  Effect of the event on the Speaker        : {xEffect}\n"
    "  Speaker's emotional reaction              : {xReact}\n\n"
    "Use this knowledge to generate a more empathetic, informed response.\n"
    "Rules: 1-3 sentences, natural and conversational, no emoji.\n"
    "Recognize and acknowledge the Speaker's feelings genuinely.\n"
    "You only need to provide the next round of response of Listener.\n\n"
    "Respond with ONLY the Listener's response text."
)


def _fmt_list(values: list) -> str:
    return ", ".join(str(v) for v in values) if values else "unknown"


async def empathy_ektc(dialogue_context: str, llm) -> dict:
    llm_calls = 0

    # ── Шаг 1: Annotator — нужно ли знание? ──────────────────────────────────
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
        # ── Шаг 2: KnowledgeGen — генерируем commonsense знание ──────────────
        knowledge_data = await llm.generate_json(
            KNOWLEDGE_SYSTEM,
            f"Dialogue:\n{dialogue_context}",
            max_tokens=256,
        )
        llm_calls += 1

        # ── Шаг 3: Reflector — проверяем консистентность ─────────────────────
        knowledge_str = (
            f"xIntent: {_fmt_list(knowledge_data.get('xIntent', []))}\n"
            f"xNeed:   {_fmt_list(knowledge_data.get('xNeed', []))}\n"
            f"xWant:   {_fmt_list(knowledge_data.get('xWant', []))}\n"
            f"xEffect: {_fmt_list(knowledge_data.get('xEffect', []))}\n"
            f"xReact:  {_fmt_list(knowledge_data.get('xReact', []))}"
        )
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
            xIntent=_fmt_list(knowledge_data.get("xIntent", [])),
            xNeed=_fmt_list(knowledge_data.get("xNeed", [])),
            xWant=_fmt_list(knowledge_data.get("xWant", [])),
            xEffect=_fmt_list(knowledge_data.get("xEffect", [])),
            xReact=_fmt_list(knowledge_data.get("xReact", [])),
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
