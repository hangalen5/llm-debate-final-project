import os
from dataset_loader import load_gsm8k_sample
from debate_engine import run_direct_baseline, run_self_consistency, run_debate

def evaluate_pilot():
    print("Loading GSM8K sample dataset...")
    samples = load_gsm8k_sample(n=3) # Chạy thử 3 mẫu trước cho nhanh
    
    print(f"Loaded {len(samples)} samples. Starting evaluation pilot...\n")
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        raw_answer = sample["answer"]
        
        # Trích xuất đáp án chuẩn từ trường 'answer' của GSM8K 
        import re
        match = re.search(r"####\s*([0-9,\.]+)", raw_answer)
        gold_answer = match.group(1).replace(",", "") if match else raw_answer
        
        print(f"--- Sample {i+1} ---")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        
        # 1. Chạy Direct Baseline
        pred_direct = run_direct_baseline(question)
        print(f"-> Direct Baseline Prediction: {pred_direct}")
        
        # 2. Chạy Self-Consistency
        pred_sc = run_self_consistency(question, k=3)
        print(f"-> Self-Consistency Prediction: {pred_sc}")
        
        # 3. Chạy Debate Engine
        pred_debate = run_debate(question, n_agents=3, n_rounds=1)
        print(f"-> Debate Engine Prediction: {pred_debate}\n")

if __name__ == "__main__":
    evaluate_pilot()