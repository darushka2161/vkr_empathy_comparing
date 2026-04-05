# architectures/empathy_loop.py
# Архитектура 3: EmpathyLoop (итеративная рефинация)
# Dialogue → [Planner] → [Generator] → validate (3 parallel) → refine until pass
# LLM-вызовов: min 5, max 11

import asyncio

_TASK_CONTEXT = (
    "This is an empathetic dialogue task. The first worker (Speaker) is given an "
    "emotion label and writes their own description of a situation when they felt "
    "that way. Then, Speaker tells their story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "The Listener's role is to recognize and acknowledge the Speaker's feelings as "
    "much as possible."
)

PLANNER_SYSTEM = (
    _TASK_CONTEXT + "\n\n"
    "You are preparing a response plan to help the Listener reply empathetically.\n"
    "Analyze the dialogue carefully and answer:\n"
    "1. emotion — the Speaker's specific feeling right now\n"
    "2. cause — the concrete event or situation that triggered it (be specific)\n"
    "3. core_need — what the Speaker most needs: "
    "validation | comfort | advice | encouragement | space_to_vent\n"
    "4. avoid — what the Listener should NOT do (e.g., give advice, minimize feelings, ask too many questions)\n"
    "5. strategy — best approach: "
    "emotional_validation | reflective_listening | gentle_curiosity | shared_humanity | encouragement\n"
    "6. key_points — 2 concrete things the response must address or reflect\n\n"
    "Respond ONLY with JSON:\n"
    '{{"emotion": "...", "cause": "...", "core_need": "...", "avoid": "...", '
    '"strategy": "...", "key_points": ["...", "..."]}}'
)

GENERATOR_SYSTEM_V1 = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot that can empathize with users. "
    "You are the Listener. Follow this response plan:\n\n"
    "  Speaker feels: {emotion}\n"
    "  Situation:     {cause}\n"
    "  Speaker needs: {core_need}\n"
    "  Avoid:         {avoid}\n"
    "  Strategy:      {strategy}\n"
    "  Must address:  {key_points}\n\n"
    "HOW TO WRITE A GENUINELY EMPATHETIC RESPONSE:\n"
    "  - Acknowledge the specific emotion and situation — not generically\n"
    "  - Sound like a caring, real person — not a template\n"
    "  - Do NOT open with \"I'm sorry to hear that\" or \"I understand\"\n"
    "  - Do NOT use hollow phrases: \"That must be tough\", \"I'm here for you\"\n"
    "  - Do NOT give unsolicited advice unless strategy is encouragement\n"
    "  - If asking a question, ask only ONE — make it feel genuinely curious\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the Listener's response text."
)

GENERATOR_REFINE = (
    _TASK_CONTEXT + "\n\n"
    "You are an empathetic conversational AI chatbot. You are the Listener.\n"
    "Your previous response was:\n"
    "  \"{previous}\"\n\n"
    "VALIDATOR FEEDBACK (address ALL of these):\n"
    "{feedback}\n\n"
    "Strategy to maintain: {strategy}\n"
    "Speaker's core need: {core_need}\n\n"
    "Write an IMPROVED response that fixes the feedback while staying genuinely empathetic.\n"
    "CRITICAL LENGTH RULE: Your response must be 1-2 sentences, maximum 15 words. "
    "Do NOT write long paragraphs.\n"
    "You only need to provide the next round of response of Listener.\n"
    "Respond with ONLY the improved Listener response text."
)

EMPATHY_VAL_SYSTEM = (
    "You are evaluating a Listener response in an empathetic dialogue. "
    "The Speaker feels {emotion}.\n\n"
    "Evaluate EMPATHY: does the response genuinely name and acknowledge the Speaker's "
    "specific feelings? Is it personal to this situation or generic?\n\n"
    "Rate 1-5:\n"
    "  5 = deeply and specifically empathetic, clearly about THIS situation\n"
    "  4 = clearly empathetic, mostly specific\n"
    "  3 = somewhat empathetic but partly generic\n"
    "  2 = barely empathetic, mostly generic\n"
    "  1 = not empathetic or dismissive\n\n"
    "Score 4+ passes. Deduct for: hollow openers (\"I'm sorry\", \"I understand\"), "
    "generic phrases (\"That must be tough\"), missing the specific emotion.\n\n"
    "Response: \"{response}\"\n\n"
    "JSON only: {{\"score\": 0, \"pass\": true, \"feedback\": \"specific actionable suggestion or null\"}}"
)

COHERENCE_VAL_SYSTEM = (
    "You are evaluating a Listener response in an empathetic dialogue.\n\n"
    "Evaluate COHERENCE: does the response directly address what the Speaker shared? "
    "Is it the right length (1-3 sentences)? Does it fit the dialogue naturally?\n\n"
    "Rate 1-5:\n"
    "  5 = directly addresses the dialogue, perfect length\n"
    "  4 = clearly relevant, appropriate length\n"
    "  3 = somewhat relevant or slightly off in length\n"
    "  2 = mostly off-topic or too long/short\n"
    "  1 = irrelevant or completely wrong length\n\n"
    "Score 4+ passes. Deduct for: ignoring key parts of the dialogue, "
    "responding to the wrong topic, being too long or too short.\n\n"
    "Response: \"{response}\"\n"
    "Context: {context}\n\n"
    "JSON only: {{\"score\": 0, \"pass\": true, \"feedback\": \"specific actionable suggestion or null\"}}"
)

SAFETY_VAL_SYSTEM = (
    "You are evaluating a Listener response in an empathetic dialogue.\n\n"
    "Evaluate QUALITY & SAFETY: check for harmful content, but also for these "
    "common empathy failures:\n"
    "  - Toxic positivity (\"everything will be fine!\", \"look on the bright side\")\n"
    "  - Empty filler standing alone (\"I'm so sorry to hear that.\" with nothing else)\n"
    "  - Unsolicited blunt advice that ignores the Speaker's feelings\n"
    "  - Minimizing language (\"at least...\", \"it could be worse\")\n"
    "  - Inappropriate humor or dismissiveness\n\n"
    "Rate 1-5:\n"
    "  5 = genuine, safe, avoids all empathy failures\n"
    "  4 = safe and mostly genuine\n"
    "  3 = some minor issues\n"
    "  2 = clear empathy failure (e.g., toxic positivity)\n"
    "  1 = harmful or deeply inappropriate\n\n"
    "Score 4+ passes.\n\n"
    "Response: \"{response}\"\n\n"
    "JSON only: {{\"score\": 0, \"pass\": true, \"feedback\": \"specific concern or null\"}}"
)


def _with_examples(system: str) -> str:
    from src.fixed_few_shot import get_few_shot_block
    return system + "\n\n" + get_few_shot_block()


async def empathy_loop(dialogue_context: str, llm, max_iter: int = 3) -> dict:
    # Step 1: Plan
    plan = await llm.generate_json(
        PLANNER_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=200,
    )
    if isinstance(plan, list):
        plan = plan[0] if plan else {}

    response = None
    all_iterations = []
    feedback_text = ""

    for iteration in range(max_iter):
        # Step 2: Generate or refine
        if iteration == 0:
            response = await llm.generate(
                _with_examples(GENERATOR_SYSTEM_V1).format(
                    emotion=plan.get("emotion", "unknown"),
                    cause=plan.get("cause", "unknown"),
                    core_need=plan.get("core_need", "validation"),
                    avoid=plan.get("avoid", "unsolicited advice"),
                    strategy=plan.get("strategy", "emotional_validation"),
                    key_points=", ".join(plan.get("key_points", [])),
                ),
                f"Dialogue:\n{dialogue_context}",
                temperature=0.1,
            )
        else:
            response = await llm.generate(
                _with_examples(GENERATOR_REFINE).format(
                    previous=response,
                    feedback=feedback_text,
                    strategy=plan.get("strategy", "emotional_validation"),
                    core_need=plan.get("core_need", "validation"),
                ),
                f"Dialogue:\n{dialogue_context}",
                temperature=0.1,
            )

        # Step 3: Validate (parallel)
        val_tasks = [
            llm.generate_json(
                EMPATHY_VAL_SYSTEM.format(
                    emotion=plan.get("emotion", "unknown"),
                    response=response,
                ),
                f"Dialogue:\n{dialogue_context}",
                max_tokens=128,
            ),
            llm.generate_json(
                COHERENCE_VAL_SYSTEM.format(
                    response=response,
                    context=dialogue_context,
                ),
                "Evaluate.",
                max_tokens=128,
            ),
            llm.generate_json(
                SAFETY_VAL_SYSTEM.format(response=response),
                "Evaluate.",
                max_tokens=128,
            ),
        ]
        v_emp, v_coh, v_saf = await asyncio.gather(*val_tasks)

        if isinstance(v_emp, list): v_emp = v_emp[0] if v_emp else {}
        if isinstance(v_coh, list): v_coh = v_coh[0] if v_coh else {}
        if isinstance(v_saf, list): v_saf = v_saf[0] if v_saf else {}

        iter_data = {
            "iteration": iteration + 1,
            "response": response,
            "empathy": v_emp,
            "coherence": v_coh,
            "safety": v_saf,
        }
        all_iterations.append(iter_data)

        # Step 4: Check if all pass
        if all([
            v_emp.get("pass", False),
            v_coh.get("pass", False),
            v_saf.get("pass", False),
        ]):
            break

        # Collect feedback for next iteration
        parts = []
        if not v_emp.get("pass"):
            parts.append(f"Empathy: {v_emp.get('feedback', 'improve empathy — be more specific')}")
        if not v_coh.get("pass"):
            parts.append(f"Coherence: {v_coh.get('feedback', 'improve relevance to this dialogue')}")
        if not v_saf.get("pass"):
            parts.append(f"Quality: {v_saf.get('feedback', 'remove filler or toxic positivity')}")
        feedback_text = "\n".join(parts)

    return {
        "response": response,
        "plan": plan,
        "iterations": all_iterations,
        "total_iterations": len(all_iterations),
        "llm_calls": 1 + len(all_iterations) * 4,
    }
