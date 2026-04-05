# load_dataset.py
# Загрузка и парсинг EmpatheticDialogues test split.
# Запуск для проверки: python load_dataset.py
#
# Требует: pip install datasets==2.19.1

from datasets import load_dataset


def prepare_examples(split: str = "test", limit: int = None) -> list[dict]:
    """
    Загружает ED и строит список примеров для экспериментов.

    Каждый пример:
      conv_id       — идентификатор диалога
      emotion       — эмоциональная метка
      context       — диалог до последней реплики listener
      gold_response — последняя реплика listener (золотой ответ)

    Args:
        split: "test", "train" или "valid"
        limit: сколько примеров вернуть (None = все)
    """
    print(f"Loading EmpatheticDialogues ({split} split)...")
    ds = load_dataset("empathetic_dialogues", split=split)

    # Группируем по conv_id
    conversations: dict = {}
    for row in ds:
        cid = row["conv_id"]
        if cid not in conversations:
            conversations[cid] = {
                "emotion": row["context"],
                "utterances": [],
            }
        conversations[cid]["utterances"].append({
            "idx": row["utterance_idx"],
            "text": row["utterance"].replace("_comma_", ","),
        })

    # Listener = чётный utterance_idx (2, 4, 6...), Speaker = нечётный (1, 3, 5...)
    examples = []
    for cid, conv in conversations.items():
        utts = sorted(conv["utterances"], key=lambda x: x["idx"])
        for i in range(len(utts) - 1, -1, -1):
            if utts[i]["idx"] % 2 == 0:  # listener
                context_utts = utts[:i]
                gold_response = utts[i]["text"]
                if context_utts:
                    context_str = "\n".join([
                        f"{'Speaker' if u['idx'] % 2 == 1 else 'Listener'}: {u['text']}"
                        for u in context_utts
                    ])
                    examples.append({
                        "conv_id": cid,
                        "emotion": conv["emotion"],
                        "context": context_str,
                        "gold_response": gold_response,
                    })
                break

    if limit:
        examples = examples[:limit]

    print(f"Prepared {len(examples)} examples")
    return examples


if __name__ == "__main__":
    print("=" * 60)
    print("ДИАГНОСТИКА ДАТАСЕТА")
    print("=" * 60)

    ds = load_dataset("empathetic_dialogues", split="test")

    print(f"\nВсего строк в test split : {len(ds)}")
    print(f"Поля (features)          : {list(ds.features.keys())}")

    row0 = ds[0]
    print(f"\n--- Первая строка ---")
    for k, v in row0.items():
        print(f"  {k}: {repr(v)}")

    # Уникальные conv_id
    conv_ids = set(r["conv_id"] for r in ds)
    print(f"\nУникальных conv_id : {len(conv_ids)}")
    print(f"Примеры conv_id    : {list(conv_ids)[:5]}")

    # Значения speaker_idx
    speaker_vals = set(r["speaker_idx"] for r in ds)
    print(f"\nЗначения speaker_idx : {speaker_vals}")

    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ПРИМЕРОВ")
    print("=" * 60)

    examples = prepare_examples(split="test")

    if examples:
        print(f"\n--- Пример 0 ---")
        ex = examples[0]
        print(f"conv_id : {ex['conv_id']}")
        print(f"emotion : {ex['emotion']}")
        print(f"context :\n{ex['context']}")
        print(f"gold    : {ex['gold_response']}")

        emotions = set(e["emotion"] for e in examples)
        print(f"\nВсего диалогов   : {len(examples)}")
        print(f"Уникальных эмоций: {len(emotions)}")
        print(f"Эмоции: {sorted(emotions)}")
    else:
        print("ОШИБКА: примеры не сформированы. Смотри диагностику выше.")
