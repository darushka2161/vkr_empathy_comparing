# architectures/empathy_mas_c.py
# Архитектура EmpathyMAS-C v2
#
# Pipeline (5 LLM-вызовов, фиксировано):
#   Dialogue
#     → [1. Emotion Agent]       LLM-вызов 1
#     → [2. FAISS Retriever]     локальный, ~5ms
#     → [3. Planner Agent]       LLM-вызов 2
#     → [4a. Gen-Comfort]        LLM-вызов 3 (параллельно с 4b)
#     → [4b. Gen-Explore]        LLM-вызов 4 (параллельно с 4a)
#     → [5. Selector Agent]      LLM-вызов 5
#     → Response

import asyncio
from architectures.empathy_chain import EMOTION_SYSTEM

_TASK_CONTEXT = (
    "This is an empathetic dialogue task. The first worker (Speaker) is given an "
    "emotion label and writes their own description of a situation when they felt "
    "that way. Then, Speaker tells their story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "The Listener's role is to recognize and acknowledge the Speaker's feelings as "
    "much as possible."
)

# ── Planner ───────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic dialogue planner. "
    "The speaker is feeling {emotion}.\n\n"
    "EXAMPLES OF GOOD EMPATHETIC RESPONSES IN SIMILAR SITUATIONS:\n"
    "1. Context: {ex1_context}\n"
    "   Response: \"{ex1_response}\"\n\n"
    "2. Context: {ex2_context}\n"
    "   Response: \"{ex2_response}\"\n\n"
    "3. Context: {ex3_context}\n"
    "   Response: \"{ex3_response}\"\n\n"
    "Create a response plan for the current dialogue. "
    "Respond ONLY with JSON:\n"
    '{{"strategy": "emotional_validation/reflective_listening/gentle_advice/'
    'shared_experience/supportive_question/encouragement", '
    '"tone": "warm/gentle/enthusiastic/serious/compassionate", '
    '"include_question": true, '
    '"key_points": ["point 1", "point 2"], '
    '"avoid": "what NOT to do", '
    '"style_notes": "brief note on style inspired by examples"}}'
)

# ── Generator-Comfort ─────────────────────────────────────────────────────────
COMFORT_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are a deeply empathetic listener. "
    "Make the speaker feel HEARD and VALIDATED.\n\n"
    "PLAN: Strategy: {strategy}, Tone: {tone}\n"
    "Points: {key_points}. Avoid: {avoid}\n\n"
    "REFERENCE EXAMPLES (match their style AND LENGTH):\n"
    "1. \"{ex1_response}\"\n"
    "2. \"{ex2_response}\"\n"
    "3. \"{ex3_response}\"\n\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "The reference examples above are typically 8-15 words. Match their length. "
    "Do NOT write long paragraphs.\n\n"
    "Approach:\n"
    "  - Acknowledge the speaker's emotional state directly\n"
    "  - Show their feelings make sense\n"
    "  - Be warm but BRIEF\n"
    "  - Do NOT open with \"I'm sorry to hear that\"\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the response text. No meta-text, no JSON."
)

# ── Generator-Explore ─────────────────────────────────────────────────────────
EXPLORE_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an engaged, curious listener. "
    "Show GENUINE INTEREST and help the speaker OPEN UP.\n\n"
    "PLAN: Strategy: {strategy}, Tone: {tone}\n"
    "Points: {key_points}. Avoid: {avoid}\n\n"
    "REFERENCE EXAMPLES (match their style AND LENGTH):\n"
    "1. \"{ex1_response}\"\n"
    "2. \"{ex2_response}\"\n"
    "3. \"{ex3_response}\"\n\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "The reference examples above are typically 8-15 words. Match their length. "
    "Do NOT write long paragraphs.\n\n"
    "Approach:\n"
    "  - Brief emotion acknowledgment (one clause, not a full sentence)\n"
    "  - One thoughtful, specific follow-up question\n"
    "  - Show you understood the context\n"
    "  - Do NOT ask generic questions like \"How are you feeling?\"\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the response text. No meta-text, no JSON."
)

# ── Selector ──────────────────────────────────────────────────────────────────
SELECTOR_SYSTEM = (
    "You are choosing the better empathetic response for a conversation. "
    "The speaker is feeling {emotion}.\n\n"
    "Response A (comfort-focused): \"{response_a}\"\n"
    "Response B (exploration-focused): \"{response_b}\"\n\n"
    "Choose the response that:\n"
    "1. Better matches the emotional needs of the speaker in this specific situation\n"
    "2. Feels more natural and genuine\n"
    "3. Would make the speaker feel more understood\n\n"
    "Consider:\n"
    "  - For intense negative emotions (devastated, terrified, guilty, sad, lonely, "
    "ashamed, anxious, afraid, disappointed, embarrassed) — "
    "comfort-focused responses usually work better.\n"
    "  - For positive or anticipatory emotions (excited, proud, surprised, hopeful, "
    "grateful, impressed, content, anticipating) — "
    "exploration-focused responses usually work better.\n"
    "  - For ambiguous or mild emotions — pick the more natural one.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{{"selected": "A or B", "reason": "one sentence explaining your choice"}}'
)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _truncate_context(context: str, max_lines: int = 3) -> str:
    lines = context.strip().split("\n")
    return "\n".join(lines[-max_lines:]) if len(lines) > max_lines else context


def _fill_retrieved(retrieved: list, emotion: str, top_k: int = 3) -> list:
    while len(retrieved) < top_k:
        retrieved.append({
            "context": "Speaker: I've been going through something difficult.",
            "gold_response": "That sounds really hard. I'm here to listen.",
            "emotion": emotion,
            "similarity": 0.0,
        })
    return retrieved


def _format_examples(retrieved: list) -> dict:
    result = {}
    for i, ex in enumerate(retrieved[:3], 1):
        result[f"ex{i}_context"] = _truncate_context(ex["context"])
        result[f"ex{i}_response"] = ex["gold_response"]
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def empathy_mas_c(dialogue_context: str, llm, retriever) -> dict:
    # ── 1. Emotion Agent (LLM-вызов 1) ───────────────────────────────────────
    emotion_data = await llm.generate_json(
        EMOTION_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=128,
    )
    if isinstance(emotion_data, list):
        emotion_data = emotion_data[0] if emotion_data else {}
    emotion = emotion_data.get("emotion", "sad")

    # ── 2. Retriever (локальный) ──────────────────────────────────────────────
    retrieved = retriever.retrieve(dialogue_context, emotion=emotion, top_k=3)
    retrieved = _fill_retrieved(retrieved, emotion, top_k=3)
    ex = _format_examples(retrieved)

    # ── 3. Planner (LLM-вызов 2) ─────────────────────────────────────────────
    plan = await llm.generate_json(
        PLANNER_SYSTEM.format(emotion=emotion, **ex),
        f"Current dialogue:\n{dialogue_context}",
        max_tokens=256,
    )
    if isinstance(plan, list):
        plan = plan[0] if plan else {}

    strategy = plan.get("strategy", "emotional_validation")
    tone = plan.get("tone", "warm")
    key_points = ", ".join(plan.get("key_points", ["acknowledge feelings"]))
    avoid = plan.get("avoid", "generic openers")
    generator_kwargs = dict(
        strategy=strategy,
        tone=tone,
        key_points=key_points,
        avoid=avoid,
        **ex,
    )

    # ── 4. Generators параллельно (LLM-вызовы 3 и 4) ─────────────────────────
    response_comfort, response_explore = await asyncio.gather(
        llm.generate(
            COMFORT_SYSTEM.format(**generator_kwargs),
            f"Dialogue:\n{dialogue_context}",
            temperature=0.1,
            max_tokens=128,
        ),
        llm.generate(
            EXPLORE_SYSTEM.format(**generator_kwargs),
            f"Dialogue:\n{dialogue_context}",
            temperature=0.1,
            max_tokens=128,
        ),
    )

    # ── 5. Selector (LLM-вызов 5) ─────────────────────────────────────────────
    selector_data = await llm.generate_json(
        SELECTOR_SYSTEM.format(
            emotion=emotion,
            response_a=response_comfort,
            response_b=response_explore,
        ),
        f"Dialogue context:\n{dialogue_context}",
        max_tokens=128,
    )
    if isinstance(selector_data, list):
        selector_data = selector_data[0] if selector_data else {}

    selected_key = selector_data.get("selected", "A").upper().strip()
    selected_response = response_comfort if selected_key == "A" else response_explore
    selected_generator = "comfort" if selected_key == "A" else "explore"

    return {
        "response": selected_response,
        "emotion": emotion_data,
        "plan": plan,
        "response_comfort": response_comfort,
        "response_explore": response_explore,
        "selector": selector_data,
        "selected_generator": selected_generator,
        "retrieved_examples": [
            {
                "context_preview": _truncate_context(r["context"], 2),
                "gold_response": r["gold_response"],
                "emotion": r["emotion"],
                "similarity": r.get("similarity", 0),
            }
            for r in retrieved
        ],
        "llm_calls": 5,
    }
