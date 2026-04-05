# architectures/empathy_chain.py
# Архитектура 1: EmpathyChain (каскадная цепочка)
# Dialogue → [Emotion] → [Cause] → [Strategy] → [Generate] → Response
# LLM-вызовов на 1 ответ: 4

# Базовый фрейминг задачи из EKTC — используется во всех агентах цепочки
_TASK_CONTEXT = (
    "This is an empathetic dialogue task. The first worker (Speaker) is given an "
    "emotion label and writes their own description of a situation when they felt "
    "that way. Then, Speaker tells their story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "The Listener's role is to recognize and acknowledge the Speaker's feelings as "
    "much as possible."
)

EMOTION_SYSTEM = f"""{_TASK_CONTEXT}

You are analyzing the dialogue as a preparation step to help the Listener respond well.
Identify the Speaker's emotional state:
1. Primary emotion (from: surprised, excited, proud, grateful, hopeful, confident, \
joyful, trusting, faithful, prepared, anticipating, content, caring, sentimental, \
nostalgic, impressed, afraid, anxious, apprehensive, embarrassed, ashamed, \
devastated, sad, disappointed, lonely, jealous, disgusted, angry, furious, \
annoyed, terrified, guilty)
2. Intensity: low, medium, or high
3. Valence: positive, negative, or neutral

Respond ONLY with JSON: {{"emotion": "...", "intensity": "...", "valence": "..."}}"""

CAUSE_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are analyzing the dialogue as a preparation step to help the Listener respond well.\n"
    "The Speaker feels {emotion} ({intensity}). Identify:\n"
    "1. The CAUSE — the specific event or situation that triggered this feeling "
    "(be concrete: what happened, what was said or lost or gained)\n"
    "2. The CORE NEED — what does the Speaker most need from the Listener right now?\n"
    "   Choose ONE: validation (feel heard and understood) | comfort (feel less alone) | "
    "advice (get practical help) | encouragement (feel hopeful) | space_to_vent (just express freely)\n"
    "3. What the Speaker is NOT asking for — e.g., if they need validation, "
    "they likely do NOT want unsolicited advice\n\n"
    'Respond ONLY with JSON: {{"cause": "...", "core_need": "...", "avoid": "..."}}'
)

STRATEGY_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are selecting the best response strategy for the Listener.\n"
    "Speaker's state: feels {emotion} ({intensity}), because: {cause}, needs: {need}\n"
    "Avoid: {avoid}\n\n"
    "Choose the single most appropriate strategy:\n"
    "- emotional_validation: directly name and validate their feeling "
    "(\"That sounds really {emotion}, and it makes complete sense\")\n"
    "- reflective_listening: mirror back what you heard to show deep understanding\n"
    "- gentle_curiosity: ask one warm, open-ended question to learn more\n"
    "- shared_humanity: briefly acknowledge that this situation is genuinely hard\n"
    "- encouragement: offer grounded, specific hope — not empty positivity\n\n"
    "Also decide:\n"
    "- tone: warm / gentle / curious / serious / compassionate\n"
    "- should the response end with a question? (true only if it feels natural)\n\n"
    'Respond ONLY with JSON: {{"strategy": "...", "tone": "...", "should_ask_question": true/false, '
    '"opening_direction": "a few words on how to open — NOT a full sentence"}}'
)

GENERATOR_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot that can empathize with users. "
    "You are the Listener. Based on the analysis below, write your next response.\n\n"
    "ANALYSIS:\n"
    "  Speaker feels: {emotion} ({intensity})\n"
    "  Situation:     {cause}\n"
    "  Speaker needs: {need}\n"
    "  Avoid:         {avoid}\n"
    "  Strategy:      {strategy}, tone: {tone}\n"
    "  End with question: {ask_question}\n"
    "  Opening direction: {opening_direction}\n\n"
    "HOW TO WRITE A GENUINELY EMPATHETIC RESPONSE:\n"
    "  - Start by acknowledging the specific emotion and situation — not generically\n"
    "  - Use natural, human language. Sound like a caring friend, not a helpdesk\n"
    "  - Do NOT open with \"I'm sorry to hear that\" — find a more specific, personal way in\n"
    "  - Do NOT give unsolicited advice unless the strategy is encouragement or gentle_curiosity\n"
    "  - Do NOT use hollow phrases: \"I understand\", \"That must be tough\", \"I'm here for you\"\n"
    "  - If asking a question, ask only ONE and make it feel genuinely curious\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the Listener's response text."
)


def _generator_system_with_examples() -> str:
    from src.fixed_few_shot import get_few_shot_block
    return GENERATOR_SYSTEM + "\n\n" + get_few_shot_block()


async def empathy_chain(dialogue_context: str, llm) -> dict:
    # Step 1: Detect emotion
    emotion_data = await llm.generate_json(
        EMOTION_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=128,
    )

    # Step 2: Identify cause and need
    cause_data = await llm.generate_json(
        CAUSE_SYSTEM.format(
            emotion=emotion_data.get("emotion", "unknown"),
            intensity=emotion_data.get("intensity", "medium"),
        ),
        f"Dialogue:\n{dialogue_context}",
        max_tokens=160,
    )

    # Step 3: Select strategy
    strategy_data = await llm.generate_json(
        STRATEGY_SYSTEM.format(
            emotion=emotion_data.get("emotion", "unknown"),
            intensity=emotion_data.get("intensity", "medium"),
            cause=cause_data.get("cause", "unknown"),
            need=cause_data.get("core_need", "validation"),
            avoid=cause_data.get("avoid", "unsolicited advice"),
        ),
        f"Dialogue:\n{dialogue_context}",
        max_tokens=160,
    )

    # Step 4: Generate response (with few-shot examples)
    response = await llm.generate(
        _generator_system_with_examples().format(
            emotion=emotion_data.get("emotion", "unknown"),
            intensity=emotion_data.get("intensity", "medium"),
            cause=cause_data.get("cause", "unknown"),
            need=cause_data.get("core_need", "validation"),
            avoid=cause_data.get("avoid", "unsolicited advice"),
            strategy=strategy_data.get("strategy", "emotional_validation"),
            tone=strategy_data.get("tone", "warm"),
            ask_question=strategy_data.get("should_ask_question", False),
            opening_direction=strategy_data.get("opening_direction", "acknowledge their feeling"),
        ),
        f"Dialogue:\n{dialogue_context}",
        max_tokens=256,
        temperature=0.1,
    )

    return {
        "response": response,
        "emotion": emotion_data,
        "intermediate": {
            "emotion": emotion_data,
            "cause": cause_data,
            "strategy": strategy_data,
        },
        "llm_calls": 4,
    }
