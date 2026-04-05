# architectures/empathy_debate.py
# Архитектура 2: EmpathyDebate (параллельные агенты + арбитр)
# Dialogue → [Emotion] → 3 agents in parallel → [Arbiter] → Best
# LLM-вызовов: 5 (1 + 3 parallel + 1)

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

# Agent A: Validates feelings — makes the Speaker feel truly heard
COMFORTER_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot. You are the Listener. "
    "The Speaker feels {emotion}.\n\n"
    "Your role: EMOTIONAL VALIDATION.\n"
    "Your goal is to make the Speaker feel genuinely heard and not alone. "
    "Name what they are feeling specifically and show it makes complete sense "
    "given their situation. Do not rush to fix or advise.\n\n"
    "HOW:\n"
    "  - Reflect the specific emotion and situation back to them\n"
    "  - Validate that their feeling is natural and understandable\n"
    "  - Use warm, personal language — not clinical or template-sounding\n"
    "  - Do NOT open with \"I'm sorry to hear that\" or \"I understand\"\n"
    "  - Do NOT offer advice or silver linings\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the Listener's response text."
)

# Agent B: Offers perspective — gently broadens the view while staying empathetic
ADVISOR_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot. You are the Listener. "
    "The Speaker feels {emotion}.\n\n"
    "Your role: EMPATHETIC PERSPECTIVE.\n"
    "First acknowledge the Speaker's feeling sincerely, then — only if it feels natural — "
    "offer a gentle reframe or a small practical thought that might actually help. "
    "The empathy must come first and feel genuine, not like a setup for advice.\n\n"
    "HOW:\n"
    "  - Open by naming or mirroring their emotional experience specifically\n"
    "  - Keep any perspective or suggestion brief and tentative (\"maybe\", \"I wonder if\")\n"
    "  - Do NOT sound like you are lecturing or minimizing their experience\n"
    "  - Do NOT open with \"I'm sorry to hear that\" or generic filler\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the Listener's response text."
)

# Agent C: Curious listener — draws the Speaker out with a thoughtful question
EXPLORER_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot. You are the Listener. "
    "The Speaker feels {emotion}.\n\n"
    "Your role: ACTIVE LISTENING with genuine curiosity.\n"
    "Show the Speaker you are truly engaged and want to understand more. "
    "Briefly acknowledge what they shared, then ask ONE open-ended question "
    "that invites them to reflect or share further. "
    "The question should feel caring and specific to their story — not generic.\n\n"
    "HOW:\n"
    "  - Open with a brief, specific acknowledgment of their situation\n"
    "  - Ask exactly ONE question — make it open-ended and genuinely curious\n"
    "  - The question should feel natural, not like a therapist's checklist\n"
    "  - Do NOT open with \"I'm sorry to hear that\" or \"How are you feeling?\"\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the Listener's response text."
)

ARBITER_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are evaluating three Listener responses to the same dialogue. "
    "The Speaker feels: {emotion}.\n\n"
    "Score each response on these four dimensions (1-5):\n"
    "  empathy     — does it genuinely name and acknowledge the Speaker's specific feelings? "
    "(5=deeply and specifically empathetic, 1=generic or dismissive)\n"
    "  relevance   — is it clearly about THIS situation, not a template response? "
    "(5=very specific, 1=could apply to any dialogue)\n"
    "  naturalness — does it sound like a real caring person, not a chatbot script? "
    "(5=very human and warm, 1=robotic or formulaic)\n"
    "  helpfulness — does it serve the Speaker's emotional need in this moment? "
    "(5=exactly what they need, 1=unhelpful or tone-deaf)\n\n"
    "SCORING PENALTIES:\n"
    "  - Opening with \"I'm sorry to hear that\" alone: -1 naturalness\n"
    "  - Generic phrases (\"I understand\", \"That must be tough\", \"I'm here for you\"): -1 empathy\n"
    "  - Unsolicited advice when Speaker needs comfort: -1 helpfulness\n\n"
    "--- Response X ---\n{response_a}\n\n"
    "--- Response Y ---\n{response_b}\n\n"
    "--- Response Z ---\n{response_c}\n\n"
    "Select the response with the highest TOTAL score. "
    "Break ties by: empathy > naturalness > relevance > helpfulness.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{{"selected": "X or Y or Z", '
    '"scores": {{"X": {{"empathy": 0, "relevance": 0, "naturalness": 0, "helpfulness": 0}}, '
    '"Y": {{"empathy": 0, "relevance": 0, "naturalness": 0, "helpfulness": 0}}, '
    '"Z": {{"empathy": 0, "relevance": 0, "naturalness": 0, "helpfulness": 0}}}}, '
    '"reason": "one sentence why this response wins"}}'
)


def _with_examples(system: str) -> str:
    from src.fixed_few_shot import get_few_shot_block
    return system + "\n\n" + get_few_shot_block()


async def empathy_debate(dialogue_context: str, llm) -> dict:
    # Step 1: Shared emotion analysis
    emotion_data = await llm.generate_json(
        EMOTION_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=128,
    )
    emotion = emotion_data.get("emotion", "unknown")

    # Step 2: Three agents in parallel (each sees fixed few-shot examples)
    tasks = [
        llm.generate(_with_examples(COMFORTER_SYSTEM).format(emotion=emotion), f"Dialogue:\n{dialogue_context}", temperature=0.1),
        llm.generate(_with_examples(ADVISOR_SYSTEM).format(emotion=emotion),   f"Dialogue:\n{dialogue_context}", temperature=0.1),
        llm.generate(_with_examples(EXPLORER_SYSTEM).format(emotion=emotion),  f"Dialogue:\n{dialogue_context}", temperature=0.1),
    ]
    response_a, response_b, response_c = await asyncio.gather(*tasks)

    # Step 3: Arbiter selects best
    arbiter_data = await llm.generate_json(
        ARBITER_SYSTEM.format(
            emotion=emotion,
            response_a=response_a,
            response_b=response_b,
            response_c=response_c,
        ),
        f"Dialogue:\n{dialogue_context}",
        max_tokens=256,
    )
    if isinstance(arbiter_data, list):
        arbiter_data = arbiter_data[0] if arbiter_data else {}

    selected_key = arbiter_data.get("selected", "X")
    candidates = {"X": response_a, "Y": response_b, "Z": response_c}

    return {
        "response": candidates.get(selected_key, response_a),
        "emotion": emotion_data,
        "all_candidates": candidates,
        "arbiter": arbiter_data,
        "llm_calls": 5,
    }
