import os
from typing import Dict, Tuple

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0,
    )


def call_deepseek_with_usage(
    prompt: str, model: str = "deepseek-flash", temperature: float = 0.7
) -> Tuple[str, Dict[str, int]]:
    empty_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that solves math problems step-by-step.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=False,
        )
        usage = response.usage
        usage_dict = {
            "prompt_tokens": int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            "completion_tokens": int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
            "total_tokens": int(usage.total_tokens) if usage and usage.total_tokens else 0,
        }
        if usage_dict["total_tokens"] == 0:
            usage_dict["total_tokens"] = usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]

        text = response.choices[0].message.content or "" if response.choices else ""
        return text, usage_dict
    except Exception as error:
        print(f"[Error] Failed to call DeepSeek API: {error}")
        return "", empty_usage


def call_deepseek(
    prompt: str, model: str = "deepseek-flash", temperature: float = 0.7
) -> str:
    text, _ = call_deepseek_with_usage(prompt, model=model, temperature=temperature)
    return text


def chat(prompt: str) -> str:
    return call_deepseek(prompt)