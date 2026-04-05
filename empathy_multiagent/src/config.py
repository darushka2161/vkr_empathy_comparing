# config.py

MODEL_REGISTRY = {
    # === GROQ (бесплатно) ===
    # Регистрация: https://console.groq.com/keys
    "llama-3.1-8b": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
        "provider": "groq",
        "size": "8B",
        "max_rpm": 30,
        "max_rpd": 14400,
        "notes": "Быстрый, бесплатный. Groq free tier.",
    },
    "qwen-3-32b": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
        "api_key_env": "GROQ_API_KEY",
        "provider": "groq",
        "size": "32B",
        "max_rpm": 30,
        "max_rpd": 14400,
        "disable_thinking": True,
        "min_max_tokens": 2048,  # thinking-блок может занять ~1500 токенов
        "notes": "Быстрый, бесплатный. Groq free tier.",
    },
    "llama-3.3-70b": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "provider": "groq",
        "size": "70B",
        "max_rpm": 30,
        "max_rpd": 14400,
        "notes": "Самая большая на Groq. Для верхней границы качества.",
    },

    # === MISTRAL API (бесплатный Experiment tier) ===
    # Регистрация: https://console.mistral.ai/api-keys
    "mistral-small": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "provider": "mistral",
        "size": "24B",
        "max_rpm": 2,
        "max_rpd": 99999,
        "notes": "Experiment tier: 2 RPM, 1B токенов/мес бесплатно.",
    },

    # === GITHUB MODELS (бесплатно) ===
    # Токен: https://github.com/settings/tokens
    "gpt-4o-mini": {
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o-mini",
        "api_key_env": "GITHUB_TOKEN",
        "provider": "github-models",
        "size": "~8B",
        "max_rpm": 10,
        "max_rpd": 150,
        "notes": "150 req/day. Для сравнения с ChatGPT.",
    },
}
