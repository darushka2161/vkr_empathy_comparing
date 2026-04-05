# architectures/empathy_rag.py
# Архитектура 4: EmpathyRAG (retrieval-augmented generation)
# Dialogue → [EmotionClassifier] → [Retriever] → [Analyzer] → [Generator]
# LLM-вызовов: 3 (retriever без LLM — только векторный поиск)

import numpy as np
from architectures.empathy_chain import EMOTION_SYSTEM


_TASK_CONTEXT = (
    "This is an empathetic dialogue task. The Speaker shares a personal situation "
    "and their feelings in a conversation with a Listener. The Listener's role is to "
    "recognize and acknowledge the Speaker's feelings as much as possible."
)

ANALYZER_SYSTEM = f"""{_TASK_CONTEXT}

You are analyzing real examples of empathetic Listener responses to a Speaker feeling {{emotion}}.

Extract what makes these responses effective:
1. Common response strategies (validation, questioning, advice, shared experience)
2. Tone patterns
3. Effective opening phrases
4. Whether follow-up questions are used
5. Response length tendency

{{examples_text}}

Respond ONLY with JSON:
{{{{
  "dominant_strategy": "most common strategy",
  "tone": "dominant tone",
  "effective_openings": ["phrase 1", "phrase 2"],
  "uses_questions": true,
  "response_length": "short/medium/long",
  "key_insight": "what makes these responses genuinely empathetic"
}}}}"""

GENERATOR_SYSTEM = f"""{_TASK_CONTEXT}

You are the Listener. The Speaker feels {{emotion}}.

PATTERNS FROM SIMILAR EMPATHETIC CONVERSATIONS:
- Strategy: {{dominant_strategy}}
- Tone: {{tone}}
- Effective openings: {{effective_openings}}
- Use follow-up question: {{uses_questions}}
- Insight: {{key_insight}}

EXAMPLES of real empathetic Listener responses for similar situations:
{{few_shot_examples}}

Now provide your response for the NEW dialogue below.
Rules:
- 1-3 sentences, natural and conversational
- Recognize and acknowledge the Speaker's feelings genuinely
- Follow the patterns above but do NOT copy examples verbatim
- No emoji
- You only need to provide the next round of response of Listener

Respond with ONLY the Listener's response text."""


def _truncate_context(context: str, max_lines: int = 4) -> str:
    lines = context.strip().split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[-max_lines:])
    return context


def _format_few_shot(retrieved: list) -> str:
    parts = []
    for i, ex in enumerate(retrieved, 1):
        short_ctx = _truncate_context(ex["context"], max_lines=3)
        parts.append(
            f"Example {i} (similarity: {ex.get('similarity', 0):.2f}):\n"
            f"Context: {short_ctx}\n"
            f"Response: {ex['gold_response']}"
        )
    return "\n\n".join(parts)


def _format_examples_for_analysis(retrieved: list) -> str:
    parts = []
    for i, ex in enumerate(retrieved, 1):
        short_ctx = _truncate_context(ex["context"], max_lines=3)
        parts.append(
            f"Example {i}:\n"
            f"Context: {short_ctx}\n"
            f"Response: {ex['gold_response']}"
        )
    return "\n\n".join(parts)


class EmpathyRetriever:
    """Векторный индекс по train-части ED, сгруппированный по эмоциям."""

    def __init__(self, train_examples: list, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.encoder = SentenceTransformer(model_name)
        self.emotion_indices = {}

        emotion_groups: dict = {}
        for ex in train_examples:
            em = ex["emotion"].lower().strip()
            emotion_groups.setdefault(em, []).append(ex)

        print(f"Building FAISS indices for {len(emotion_groups)} emotions...")
        for emotion, examples in emotion_groups.items():
            contexts = [ex["context"] for ex in examples]
            embeddings = self.encoder.encode(
                contexts, show_progress_bar=False, batch_size=64
            )
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self.emotion_indices[emotion] = (index, examples)

    def retrieve(self, query_context: str, emotion: str, top_k: int = 3) -> list:
        import faiss

        emotion = emotion.lower().strip()
        if emotion not in self.emotion_indices:
            return self._fallback_retrieve(query_context, top_k)

        index, examples = self.emotion_indices[emotion]
        query_emb = self.encoder.encode([query_context])
        query_emb = np.array(query_emb, dtype="float32")
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                ex = examples[idx]
                results.append({
                    "context": ex["context"],
                    "gold_response": ex["gold_response"],
                    "emotion": ex["emotion"],
                    "similarity": float(score),
                })
        return results

    def _fallback_retrieve(self, query_context: str, top_k: int = 3) -> list:
        import faiss

        query_emb = self.encoder.encode([query_context])
        query_emb = np.array(query_emb, dtype="float32")
        faiss.normalize_L2(query_emb)
        all_results = []
        for emotion, (index, examples) in self.emotion_indices.items():
            scores, indices = index.search(query_emb, 1)
            if indices[0][0] >= 0:
                ex = examples[indices[0][0]]
                all_results.append((float(scores[0][0]), ex))
        all_results.sort(key=lambda x: -x[0])
        return [
            {
                "context": ex["context"],
                "gold_response": ex["gold_response"],
                "emotion": ex["emotion"],
                "similarity": score,
            }
            for score, ex in all_results[:top_k]
        ]

    def save(self, path: str):
        import pickle
        import os
        import faiss

        os.makedirs(path, exist_ok=True)
        for emotion, (index, examples) in self.emotion_indices.items():
            faiss.write_index(index, os.path.join(path, f"{emotion}.faiss"))
        with open(os.path.join(path, "examples.pkl"), "wb") as f:
            pickle.dump(
                {em: exs for em, (_, exs) in self.emotion_indices.items()}, f
            )
        print(f"Retriever saved to: {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2"):
        import pickle
        import os
        import faiss
        from sentence_transformers import SentenceTransformer

        with open(os.path.join(path, "examples.pkl"), "rb") as f:
            emotion_examples = pickle.load(f)
        obj = cls.__new__(cls)
        obj.encoder = SentenceTransformer(model_name)
        obj.emotion_indices = {}
        for emotion, examples in emotion_examples.items():
            index = faiss.read_index(os.path.join(path, f"{emotion}.faiss"))
            obj.emotion_indices[emotion] = (index, examples)
        print(f"Retriever loaded from: {path}")
        return obj


async def empathy_rag(dialogue_context: str, llm, retriever: EmpathyRetriever, top_k: int = 7) -> dict:
    # AGENT 1: Emotion Classifier
    emotion_data = await llm.generate_json(
        EMOTION_SYSTEM,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=128,
    )
    predicted_emotion = emotion_data.get("emotion", "sad")

    # AGENT 2: Retriever (no LLM)
    retrieved = retriever.retrieve(
        query_context=dialogue_context,
        emotion=predicted_emotion,
        top_k=top_k,
    )
    # Дополняем до top_k если не нашли достаточно
    while len(retrieved) < top_k:
        retrieved.append({
            "context": "No similar example found",
            "gold_response": "I understand how you feel.",
            "emotion": predicted_emotion,
            "similarity": 0.0,
        })

    # AGENT 3: Example Analyzer
    examples_text = _format_examples_for_analysis(retrieved)
    analysis = await llm.generate_json(
        ANALYZER_SYSTEM.format(emotion=predicted_emotion, examples_text=examples_text),
        "Analyze these empathetic response examples.",
        max_tokens=256,
    )
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else {}

    # AGENT 4: Response Generator
    few_shot_text = _format_few_shot(retrieved)
    response = await llm.generate(
        GENERATOR_SYSTEM.format(
            emotion=predicted_emotion,
            dominant_strategy=analysis.get("dominant_strategy", "emotional_validation"),
            tone=analysis.get("tone", "warm"),
            effective_openings=", ".join(analysis.get("effective_openings", ["I can understand"])),
            uses_questions=analysis.get("uses_questions", True),
            key_insight=analysis.get("key_insight", "Show genuine understanding"),
            few_shot_examples=few_shot_text,
        ),
        f"Dialogue:\n{dialogue_context}",
        temperature=0.3,
        max_tokens=256,
    )

    return {
        "response": response,
        "emotion": emotion_data,
        "retrieved_examples": [
            {
                "context_preview": _truncate_context(r["context"], 2),
                "gold_response": r["gold_response"],
                "emotion": r["emotion"],
                "similarity": r.get("similarity", 0),
            }
            for r in retrieved
        ],
        "analysis": analysis,
        "llm_calls": 3,
    }
