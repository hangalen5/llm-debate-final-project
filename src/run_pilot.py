import re
from dataset_loader import load_gsm8k_sample
from debate_engine import run_debate_with_judge

def evaluate_pilot():
    print("Loading GSM8K dataset samples...")
    n_samples = 10  # Scale up evaluation size to 10 samples for robust reporting
    samples = load_gsm8k_sample(n=n_samples)
    
    print(f"Loaded {len(samples)} samples. Starting evaluation with Judge format...\n")
    
    correct_count = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        raw_answer = sample["answer"]
        
        # Extract ground truth gold answer from GSM8K format
        match = re.search(r"####\s*([0-9,\.]+)", raw_answer)
        gold_answer = match.group(1).replace(",", "") if match else raw_answer
        
        print("========================================")
        print(f"=== Sample {i+1}/{len(samples)} ===")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        print("========================================")
        
        # Run Multi-Agent Debate with Judge
        debate_res = run_debate_with_judge(question, n_rounds=1)
        
        print(f"\n[Debater A Final]:\n{debate_res['debater_a_final']}")
        print(f"\n[Debater B Final]:\n{debate_res['debater_b_final']}")
        print("\n[Judge JSON Decision]:")
        print(debate_res['judge'])
        
        predicted_answer = debate_res['prediction']
        print(f"\n-> Final Extracted Prediction: {predicted_answer}")
        
        is_ok = (predicted_answer == gold_answer)
        if is_ok:
            correct_count += 1
            
        print(f"-> Result: {'✓ CORRECT' if is_ok else '✗ INCORRECT'}\n")

    print("========================================")
    print("           EVALUATION SUMMARY           ")
    print("========================================")
    print(f"Total Evaluated: {len(samples)}")
    print(f"Accuracy: {correct_count}/{len(samples)} ({correct_count/len(samples)*100:.1f}%)")
    print("========================================")

if __name__ == "__main__":
    evaluate_pilot()