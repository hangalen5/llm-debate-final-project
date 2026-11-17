import os
from openai import OpenAI

def get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Environment variable DEEPSEEK_API_KEY is not set!")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0 # Thêm timeout để tránh treo kết nối
    )

def call_deepseek(prompt: str) -> str:
    """Gọi DeepSeek API với model 'deepseek-chat'"""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves math problems step-by-step."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Error] Failed to call DeepSeek API: {e}")
        return ""

def chat(prompt: str) -> str:
    """Hàm wrapper tương thích với debate engine"""
    return call_deepseek(prompt)