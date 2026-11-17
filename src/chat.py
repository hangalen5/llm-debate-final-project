import os

from openai import OpenAI


def get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0,
    )


def call_deepseek(
    prompt: str, model: str = "deepseek-flash", temperature: float = 0.7
) -> str:
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves math problems step-by-step."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=False,
        )
        usage = response.usage
        if usage:
            print(
                f"[API Token Usage] Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, Total: {usage.total_tokens}"
            )
        return response.choices[0].message.content or ""
    except Exception as error:
        print(f"[Error] Failed to call DeepSeek API: {error}")
        return ""


def chat(prompt: str) -> str:
    return call_deepseek(prompt)
