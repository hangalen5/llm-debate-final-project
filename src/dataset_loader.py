if __package__:
    from .answers import extract_gsm8k_answer
else:
    from answers import extract_gsm8k_answer

def load_gsm8k_sample(n: int = 3, split: str = "test"):
    """
    Load a sample of GSM8K dataset from Hugging Face.
    Uses 'openai/gsm8k' with the 'main' configuration.
    """
    from datasets import load_dataset

    if n < 1:
        raise ValueError("n must be positive")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    
    samples = []
    for i in range(min(n, len(ds))):
        samples.append({
            "question_id": f"gsm8k:{split}:{i}",
            "question": ds[i]["question"],
            "answer": ds[i]["answer"]
        })
    return samples
