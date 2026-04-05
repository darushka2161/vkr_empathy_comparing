# metrics.py

import numpy as np
from collections import Counter


# --- BLEU (corpus-level, ×100) ---

def compute_bleu(hypotheses: list, references: list) -> dict:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    smooth = SmoothingFunction().method1
    refs = [[word_tokenize(ref.lower(), preserve_line=True)] for ref in references]
    hyps = [word_tokenize(hyp.lower(), preserve_line=True) for hyp in hypotheses]
    return {
        "BLEU-1": round(corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth) * 100, 2),
        "BLEU-2": round(corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth) * 100, 2),
        "BLEU-3": round(corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth) * 100, 2),
        "BLEU-4": round(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100, 2),
    }


# --- ROUGE-1/2/L (F1, ×100) ---

def compute_rouge(hypotheses: list, references: list) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "ROUGE-1": round(np.mean(r1) * 100, 2),
        "ROUGE-2": round(np.mean(r2) * 100, 2),
        "ROUGE-L": round(np.mean(rl) * 100, 2),
    }


# --- BERTScore (не умножаем на 100) ---

def compute_bertscore(hypotheses: list, references: list) -> dict:
    from bert_score import score as bscore
    P, R, F1 = bscore(
        hypotheses,
        references,
        lang="en",
        verbose=False,
    )
    return {
        "BERTScore-P": round(P.mean().item(), 6),
        "BERTScore-R": round(R.mean().item(), 6),
        "BERTScore-F": round(F1.mean().item(), 6),
    }


# --- Distinct-1/2 (×100 для сопоставимости с лидербордом) ---

def compute_distinct(hypotheses: list) -> dict:
    from nltk.tokenize import word_tokenize

    def dist_n(texts, n):
        ngrams = {}
        total = 0
        for t in texts:
            tokens = word_tokenize(t.lower(), preserve_line=True)
            for i in range(len(tokens) - n + 1):
                key = tuple(tokens[i : i + n])
                ngrams[key] = 1
                total += 1
        return len(ngrams) / (total + 1e-16)

    return {
        "Dist-1": round(dist_n(hypotheses, 1) * 100, 2),
        "Dist-2": round(dist_n(hypotheses, 2) * 100, 2),
    }


# --- Emotion Accuracy ---

def compute_accuracy(predicted: list, gold: list) -> dict:
    from sklearn.metrics import accuracy_score
    pred = [e.lower().strip() for e in predicted]
    gold_clean = [e.lower().strip() for e in gold]
    return {"Accuracy (%)": round(accuracy_score(gold_clean, pred) * 100, 2)}


# --- Средняя длина ---

def compute_avg_length(hypotheses: list) -> dict:
    lengths = [len(h.split()) for h in hypotheses]
    return {"AvgLen": round(np.mean(lengths), 1)}


# --- Cost (кол-во вызовов) ---

def compute_cost(results: list) -> dict:
    calls = [r.get("llm_calls", 1) for r in results]
    return {
        "Total LLM calls": sum(calls),
        "Avg calls/example": round(np.mean(calls), 1),
    }


# --- RAG-специфичные метрики ---

def compute_rag_metrics(results: list) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    similarities, novelties = [], []
    for r in results:
        retrieved = r.get("intermediate", {}).get("retrieved_examples", [])
        for ex in retrieved:
            if ex.get("similarity", 0) > 0:
                similarities.append(ex["similarity"])
        gen = r.get("generated_response", "").lower().split()
        if gen and retrieved:
            max_bleu = max(
                (
                    sentence_bleu(
                        [ex["gold_response"].lower().split()], gen,
                        smoothing_function=smooth,
                    )
                    for ex in retrieved if ex.get("gold_response")
                ),
                default=0,
            )
            novelties.append(1.0 - max_bleu)
    return {
        "Avg retrieval similarity": round(np.mean(similarities), 4) if similarities else 0,
        "Response novelty": round(np.mean(novelties), 4) if novelties else 0,
    }


# --- Собрать все метрики разом ---

def compute_all_metrics(
    hypotheses: list,
    references: list,
    predicted_emotions: list = None,
    gold_emotions: list = None,
    results: list = None,
    skip_bertscore: bool = False,
) -> dict:
    """Считает все метрики и возвращает единый dict."""
    metrics = {}
    metrics.update(compute_bleu(hypotheses, references))
    metrics.update(compute_rouge(hypotheses, references))
    if not skip_bertscore:
        metrics.update(compute_bertscore(hypotheses, references))
    metrics.update(compute_distinct(hypotheses))
    metrics.update(compute_avg_length(hypotheses))

    if predicted_emotions and gold_emotions:
        metrics.update(compute_accuracy(predicted_emotions, gold_emotions))

    if results:
        metrics.update(compute_cost(results))

    return metrics
