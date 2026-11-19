import json
import re
from dataset_loader import load_gsm8k_sample
from debate_engine import run_debate_with_judge

def evaluate_pilot():
    print("Loading GSM8K dataset samples...")
    n_samples = 10  # Number of samples to evaluate
    samples = load_gsm8k_sample(n=n_samples)
    
    print(f"Loaded {len(samples)} samples. Starting evaluation with Judge format...\n")
    
    correct_count = 0
    results_list = []  # List to store detailed results for JSON export
    
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
            
        result_status = 'CORRECT' if is_ok else 'INCORRECT'
        print(f"-> Result: {'✓ ' + result_status if is_ok else '✗ ' + result_status}\n")
        
        # Append structured data for JSON logging
        results_list.append({
            "sample_id": i + 1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "status": result_status,
            "debater_a_final": debate_res["debater_a_final"],
            "debater_b_final": debate_res["debater_b_final"],
            "judge_decision": debate_res["judge"]
        })

    # Summary metrics
    accuracy = correct_count / len(samples) * 100
    summary = {
        "total_evaluated": len(samples),
        "correct_count": correct_count,
        "accuracy_percent": accuracy,
        "results": results_list
    }

    # Save results directly to the project root directory as evaluation_results.json
    output_filename = "../evaluation_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"-> Saved detailed results to project root ({output_filename}) successfully!")

    print("========================================")
    print("           EVALUATION SUMMARY           ")
    print("========================================")
    print(f"Total Evaluated: {len(samples)}")
    print(f"Accuracy: {correct_count}/{len(samples)} ({accuracy:.1f}%)")
    print(f"Results Logged to: evaluation_results.json (at project root)")
    print("========================================")

if __name__ == "__main__":
    evaluate_pilot()