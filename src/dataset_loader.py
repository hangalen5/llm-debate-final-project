from datasets import load_dataset

def load_gsm8k_sample(n: int = 3, split: str = "test"):
    """
    Load a sample of GSM8K dataset from Hugging Face.
    Uses 'openai/gsm8k' with the 'main' configuration.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    
    samples = []
    for i in range(min(n, len(ds))):
        samples.append({
            "question": ds[i]["question"],
            "answer": ds[i]["answer"]
        })
    return samples

def extract_gsm8k_answer(response_text: str) -> str:
    """
    Extract the final numeric answer from the model's response text.
    Looks for patterns like 'Answer: <number>'.
    """
    import re
    # Tìm kiếm pattern "Answer: <số>" hoặc các biến thể tương tự
    match = re.search(r"Answer:\s*([0-9,\.]+)", response_text, re.IGNORECASE)
    if match:
        ans = match.group(1).replace(",", "")
        return ans
    
    # Dự phòng: tìm số cuối cùng xuất hiện trong văn bản nếu không tìm thấy từ khóa Answer:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
    if numbers:
        return numbers[-1]
        
    return ""