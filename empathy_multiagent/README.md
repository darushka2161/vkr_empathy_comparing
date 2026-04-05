# Empathy Multiagent

Мультиагентные архитектуры для генерации эмпатичных ответов.
Бенчмарк: **EmpatheticDialogues** (Rashkin et al., 2019), test split (~2 547 диалогов).

---

## Структура проекта

```
empathy_multiagent/
├── .env                        # API ключи (НЕ в git!)
├── .env.example                # Шаблон для .env
├── .gitignore
├── run_experiment.py           # Главный скрипт запуска
├── requirements.txt
│
├── src/                        # Основные модули
│   ├── config.py               # MODEL_REGISTRY — все доступные модели
│   ├── llm_factory.py          # Универсальный LLM клиент (OpenAI-compatible)
│   ├── load_dataset.py         # Загрузка EmpatheticDialogues
│   └── metrics.py              # BLEU, ROUGE, BERTScore, Distinct, Accuracy
│
├── architectures/
│   ├── empathy_chain.py        # Архитектура 1: каскадная цепочка (4 вызова)
│   ├── empathy_debate.py       # Архитектура 2: параллельные агенты + арбитр (5 вызовов)
│   ├── empathy_loop.py         # Архитектура 3: итеративная рефинация (5–11 вызовов)
│   └── empathy_rag.py          # Архитектура 4: RAG (3 вызова + векторный поиск)
│
├── analysis/
│   ├── compare_results.py      # Сводная таблица и графики по всем outputs/
│   └── recompute_metrics.py    # Пересчёт метрик без повторного запуска архитектур
│
└── outputs/                    # Результаты экспериментов (создаётся автоматически)
    ├── <model>_<arch>.json
    ├── summary.csv
    └── plots/
        └── <metric>.png
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

Для BERTScore нужен PyTorch. Если GPU нет — добавь `--no-bertscore` при запуске (см. ниже).

### 2. Настройка API ключей

```bash
cp .env.example .env
```

Открой `.env` и вставь свои ключи.
**Для старта достаточно только `GROQ_API_KEY`** (работает с `llama-3.1-8b` и `llama-3.3-70b`).

Получить Groq API key: [https://console.groq.com](https://console.groq.com) → API Keys (бесплатно).

### 3. Проверка подключения

```bash
python -c "from src.llm_factory import LLMFactory; import asyncio; asyncio.run(LLMFactory('llama-3.1-8b').generate('Hi', 'Say hello'))"
```

---

## Запуск экспериментов

### Синтаксис

```bash
python run_experiment.py --model <MODEL> --arch <ARCH> [--limit N] [--no-bertscore]
```

| Аргумент | Описание | Пример |
|---|---|---|
| `--model` | Ключ модели из `src/config.py` | `llama-3.1-8b` |
| `--arch` | Архитектура | `empathy_chain` |
| `--limit N` | Сколько диалогов прогнать (по умолчанию — все ~2547) | `--limit 50` |
| `--no-bertscore` | Пропустить BERTScore (быстрее, нет GPU) | флаг |

### Примеры

```bash
# Быстрый тест — убедиться, что всё работает
python run_experiment.py --model llama-3.1-8b --arch empathy_chain --limit 10 --no-bertscore

# Базовые подходы (baseline)
python run_experiment.py --model llama-3.1-8b --arch empathy_zero_shot --limit 50
python run_experiment.py --model llama-3.1-8b --arch empathy_few_shot --limit 50
python run_experiment.py --model llama-3.1-8b --arch empathy_ektc --limit 50

# Отладка на 50 диалогах
python run_experiment.py --model llama-3.1-8b --arch empathy_chain --limit 50
python run_experiment.py --model llama-3.1-8b --arch empathy_debate --limit 50
python run_experiment.py --model llama-3.1-8b --arch empathy_loop --limit 50
python run_experiment.py --model llama-3.1-8b --arch empathy_rag --limit 50

# Полный прогон на всех 2547 диалогах
python run_experiment.py --model llama-3.1-8b --arch empathy_zero_shot
python run_experiment.py --model llama-3.1-8b --arch empathy_few_shot
python run_experiment.py --model llama-3.1-8b --arch empathy_ektc
python run_experiment.py --model llama-3.1-8b --arch empathy_chain
python run_experiment.py --model llama-3.1-8b --arch empathy_debate
python run_experiment.py --model llama-3.1-8b --arch empathy_loop
python run_experiment.py --model llama-3.1-8b --arch empathy_rag
python run_experiment.py --model llama-3.1-8b --arch empathy_mas_c

# Без BERTScore (если нет GPU или нужно быстро)
python run_experiment.py --model llama-3.1-8b --arch empathy_chain --no-bertscore

# Другая модель
python run_experiment.py --model gpt-4o-mini --arch empathy_chain --limit 100
```

### Прогнать все комбинации (bash)

```bash
for model in llama-3.1-8b qwen-3-32b llama-3.3-70b mistral-small gpt-4o-mini; do
  for arch in empathy_zero_shot empathy_few_shot empathy_ektc empathy_chain empathy_debate empathy_loop empathy_rag empathy_mas_c; do
    python run_experiment.py --model $model --arch $arch --no-bertscore
  done
done
```

---

## Анализ результатов

### Сводная таблица и графики

```bash
python analysis/compare_results.py
```

Выводит таблицу в консоль, сохраняет `outputs/summary.csv` и PNG-графики в `outputs/plots/` (по одному на каждую метрику).

### Пересчёт метрик без повторного запуска

Если изменилась реализация метрик — пересчитать по уже готовым результатам:

```bash
# Пересчитать все файлы в outputs/
python analysis/recompute_metrics.py

# Без BERTScore (быстро)
python analysis/recompute_metrics.py --no-bertscore

# Только конкретные файлы
python analysis/recompute_metrics.py --files outputs/gpt-4o-mini_empathy_chain.json outputs/llama-3.1-8b_empathy_rag.json
```

---

## Доступные модели

| Ключ (`--model`) | Провайдер | Размер | Бесплатно | Нужен ключ |
|---|---|---|---|---|
| `llama-3.1-8b` | Groq | 8B | ✅ (лимиты по RPM) | `GROQ_API_KEY` |
| `llama-3.3-70b` | Groq | 70B | ✅ (лимиты по RPM) | `GROQ_API_KEY` |
| `llama-3.3-70b-cerebras` | Cerebras | 70B | ✅ (1M токенов/день) | `CEREBRAS_API_KEY` |
| `gemini-2.5-flash` | Google AI Studio | — | ✅ (10 RPM / 250 RPD) | `GEMINI_API_KEY` |
| `gemini-2.5-pro` | Google AI Studio | — | ✅ (5 RPM / 100 RPD) | `GEMINI_API_KEY` |
| `mistral-small` | Mistral API | 24B | ✅ (2 RPM, 1B токенов/мес) | `MISTRAL_API_KEY` |
| `gpt-4o-mini` | GitHub Models | ~8B | ✅ (лимиты по tier) | `GITHUB_TOKEN` |
| `deepseek-v3` | OpenRouter | 685B MoE | ⚠️ платная | `OPENROUTER_API_KEY` |
| `qwen-2.5-7b` | Together AI | 7B | ⚠️ кредиты при регистрации | `TOGETHER_API_KEY` |
| `qwen-2.5-14b` | Together AI | 14B | ⚠️ кредиты при регистрации | `TOGETHER_API_KEY` |
| `qwen-2.5-32b` | Together AI | 32B | ⚠️ кредиты при регистрации | `TOGETHER_API_KEY` |

> ✅ — постоянный бесплатный tier · ⚠️ — стартовые кредиты или ограниченный доступ

---

## Доступные архитектуры

| Ключ (`--arch`) | Описание | LLM-вызовов/диалог |
|---|---|---|
| `empathy_zero_shot` | Прямой вызов с системным промптом EKTC (baseline) | 1 |
| `empathy_few_shot` | 3 случайных примера из train в промпте (baseline) | 1 |
| `empathy_ektc` | Annotator → KnowledgeGen → Reflector → Generator (COLING 2025) | 2–4 |
| `empathy_chain` | Каскад: эмоция → причина → стратегия → ответ | 4 |
| `empathy_debate` | Параллельно 3 агента, арбитр выбирает лучший | 5 |
| `empathy_loop` | Генератор + 3 валидатора, итерации до качества | 5–11 |
| `empathy_rag` | Векторный поиск по train-примерам + анализ + генерация | 3 |
| `empathy_mas_c` | LLM emotion + RAG + Planner + 2×Generator (parallel) + Selector + Checker | 6 |

---

## Метрики

| Метрика | Описание |
|---|---|
| BLEU-1/2/3/4 | N-gram precision ×100, со сглаживанием (method1) |
| ROUGE-1/2/L | F1 overlap ×100, без стемминга |
| BERTScore-P/R/F | Семантическое сходство (roberta-large, lang=en) |
| Dist-1/2 | Разнообразие ответов ×100 |
| AvgLen | Средняя длина ответа (слов) |
| Accuracy (%) | Точность определения эмоции |
| Avg calls/example | Среднее число LLM-вызовов на диалог |
| Avg latency ms | Среднее время на диалог |
| Errors | Число ошибок |

---

## Лимиты Groq (бесплатный tier)

| Архитектура | Вызовов на диалог | Всего (2547 диал.) | Время (30 rpm) |
|---|---|---|---|
| empathy_chain | 4 | 10 188 | ~340 мин |
| empathy_debate | 5 | 12 735 | ~425 мин |
| empathy_loop | 5–11 (ср. ~7) | ~17 829 | ~594 мин |
| empathy_rag | 3 | 7 641 | ~255 мин |

Groq: 30 req/min, 14 400 req/day.
Рекомендация: начни с `--limit 50`, потом `--limit 500`, затем полный прогон.

---

## Добавить новую модель

В `src/config.py` добавить запись в `MODEL_REGISTRY`:

```python
"my-model": {
    "base_url": "https://api.example.com/v1",
    "model": "exact-model-name",
    "api_key_env": "MY_API_KEY",
    "provider": "example",
    "size": "7B",
    "max_rpm": 30,
    "max_rpd": 10000,
    "notes": "описание",
},
```

Затем: `python run_experiment.py --model my-model --arch empathy_chain --limit 10`
