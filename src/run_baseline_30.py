import json
import os
import re
from dataset_loader import load_gsm8k_sample
from chat import call_deepseek_with_usage

INPUT_PRICE_RMB_PER_1M = float(os.getenv("INPUT_PRICE_RMB_PER_1M", "2.0"))
OUTPUT_PRICE_RMB_PER_1M = float(os.getenv("OUTPUT_PRICE_RMB_PER_1M", "3.0"))


def compute_cost_rmb(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000.0) * INPUT_PRICE_RMB_PER_1M + (
        completion_tokens / 1_000_000.0
    ) * OUTPUT_PRICE_RMB_PER_1M


def extract_prediction(text: str) -> str:
    boxed_match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed_match:
        boxed_content = boxed_match.group(1).replace("{,}", "").replace(",", "")
        nums_in_box = re.findall(r"[0-9]+(?:\.[0-9]+)?", boxed_content)
        if nums_in_box:
            return nums_in_box[-1]

    bold_match = re.search(r"\*\*\$?([0-9,\.]+)", text)
    if bold_match:
        return bold_match.group(1).replace(",", "")

    numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", text)
    return numbers[-1].replace(",", "") if numbers else ""


def evaluate_baseline():
    print("Loading GSM8K dataset samples...")
    n_samples = 30
    samples = load_gsm8k_sample(n=n_samples)

    print(f"Loaded {len(samples)} samples. Starting Baseline evaluation...\n")

    correct_count = 0
    results_list = []
    total_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for i, sample in enumerate(samples):
        question = sample["question"]
        raw_answer = sample["answer"]

        match = re.search(r"####\s*([0-9,\.]+)", raw_answer)
        gold_answer = match.group(1).replace(",", "") if match else raw_answer

        prompt = (
            "Answer the following math question step by step "
            f"and clearly state your final numerical answer.\n\nQuestion: {question}"
        )
        reply, u = call_deepseek_with_usage(prompt)

        p_tok = int(u.get("prompt_tokens", 0))
        c_tok = int(u.get("completion_tokens", 0))
        t_tok = int(u.get("total_tokens", p_tok + c_tok))
        s_cost = compute_cost_rmb(p_tok, c_tok)

        total_usage["calls"] += 1
        total_usage["prompt_tokens"] += p_tok
        total_usage["completion_tokens"] += c_tok
        total_usage["total_tokens"] += t_tok

        predicted_answer = extract_prediction(reply)
        is_ok = (predicted_answer == gold_answer)
        if is_ok:
            correct_count += 1

        result_status = "CORRECT" if is_ok else "INCORRECT"
        print(
            f"[{i+1}/{len(samples)}] {result_status} | "
            f"pred={predicted_answer} vs gold={gold_answer} | "
            f"tokens={t_tok} | cost={s_cost:.6f} RMB"
        )

        results_list.append({
            "sample_id": i + 1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "status": result_status,
            "reply": reply,
            "usage": {
                "calls": 1,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
                "cost_rmb": round(s_cost, 6),
            },
        })

    accuracy = correct_count / len(samples) * 100
    total_cost_rmb = compute_cost_rmb(
        total_usage["prompt_tokens"], total_usage["completion_tokens"]
    )

    payload = {
        "total_evaluated": len(samples),
        "correct_count": correct_count,
        "accuracy_percent": accuracy,
        "usage": {
            **total_usage,
            "cost_rmb": round(total_cost_rmb, 6),
        },
        "results": results_list,
    }

    output_path = "baseline_results_30.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print("\n========================================")
    print("           BASELINE SUMMARY             ")
    print("========================================")
    print(f"Total Evaluated: {len(samples)}")
    print(f"Accuracy: {correct_count}/{len(samples)} ({accuracy:.1f}%)")
    print(f"API Calls: {total_usage['calls']}")
    print(f"Prompt Tokens: {total_usage['prompt_tokens']:,}")
    print(f"Completion Tokens: {total_usage['completion_tokens']:,}")
    print(f"Total Tokens: {total_usage['total_tokens']:,}")
    print(f"Total Cost (RMB): {total_cost_rmb:.4f}")
    print(f"Results Logged to: {output_path}")
    print("========================================")


if __name__ == "__main__":
    evaluate_baseline()
