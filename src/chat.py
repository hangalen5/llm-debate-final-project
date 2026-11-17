import os
import json
from openai import OpenAI

# Initialize OpenAI client connected to DeepSeek API
# (DeepSeek uses OpenAI-compatible standard)
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def call_deepseek(prompt: str, model: str = "deepseek-chat", temperature: float = 0.7) -> str:
    """
    Function to call DeepSeek API to get a response from the LLM.
    Can be extended with caching if needed.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and rigorous debater assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=False
        )
        
        # Extract the response content
        answer = response.choices[0].message.content
        
        # Log token usage for cost control (project requirement)
        usage = response.usage
        if usage:
            print(f"[API Token Usage] Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
        return answer
    except Exception as e:
        print(f"[Error] Failed to call DeepSeek API: {e}")
        return ""
