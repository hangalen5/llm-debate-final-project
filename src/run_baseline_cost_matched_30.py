import json
import os
import re
from collections import Counter
from dataset_loader import load_gsm8k_sample
from chat import call_deepseek_with_usage

INPUT_PRICE_RMB_PER_1M = float(os.getenv("INPUT_PRICE_RMB_PER_1M", "2.0"))
OUTPUT_PRICE_RMB_PER_1M = float(os.getenv("OUTPUT_PRICE_RMB_PER_1M", "3.0"))
TARGET_COST_RMB = float(os.getenv("TARGET_COST_RMB", "0.2880"))
MAX_PASSES = int(os.getenv("MAX_PASSES", "10"))


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


def majority_vote(preds: list[str]) -> str:
    valid = [p for p in preds if p != ""]
    if not valid:
        return ""
    counts = Counter(valid)
    best_count = max(counts.values())
    # Giữ đáp án xuất hiện sớm nhất trong số các đáp án đạt số phiếu cao nhất
    for p in valid:
        if counts[p] == best_count:
            return p
    return valid[0]


def evaluate_baseline_cost_matched():
    print("Loading GSM8K dataset samples...")
    samples = load_gsm8k_sample(n=30)
    n_samples = len(samples)

    per_sample_replies: list[list[str]] = [[] for _ in range(n_samples)]
    per_sample_preds: list[list[str]] = [[] for _ in range(n_samples)]

    total_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    prompt_template = (
        "Answer the following math question step by step "
        "and clearly state your final numerical answer.\n\nQuestion: {question}"
    )

    passes_completed = 0
    stop_reason = "max_passes_reached"

    for p in range(1, MAX_PASSES + 1):
        pass_calls = 0
        pass_prompt_tok = 0
        pass_completion_tok = 0
        pass_total_tok = 0

        for i, sample in enumerate(samples):
            current_cost = compute_cost_rmb(
                total_usage["prompt_tokens"], total_usage["completion_tokens"]
            )
            if current_cost >= TARGET_COST_RMB:
                stop_reason = "target_cost_reached"
                break

            question = sample["question"]
            reply, u = call_deepseek_with_usage(prompt_template.format(question=question))

            p_tok = int(u.get("prompt_tokens", 0))
            c_tok = int(u.get("completion_tokens", 0))
            t_tok = int(u.get("total_tokens", p_tok + c_tok))

            total_usage["calls"] += 1
            total_usage["prompt_tokens"] += p_tok
            total_usage["completion_tokens"] += c_tok
            total_usage["total_tokens"] += t_tok

            pass_calls += 1
            pass_prompt_tok += p_tok
            pass_completion_tok += c_tok
            pass_total_tok += t_tok

            pred = extract_prediction(reply)
            per_sample_replies[i].append(reply)
            per_sample_preds[i].append(pred)

        if pass_calls > 0:
            passes_completed = p

        pass_cost = compute_cost_rmb(pass_prompt_tok, pass_completion_tok)
        running_cost = compute_cost_rmb(
            total_usage["prompt_tokens"], total_usage["completion_tokens"]
        )
        print(
            f"[Pass {p}] calls={pass_calls} | pass_cost={pass_cost:.4f} RMB | "
            f"cumulative_cost={running_cost:.4f} / {TARGET_COST_RMB:.4f} RMB"
        )

        if stop_reason == "target_cost_reached" or running_cost >= TARGET_COST_RMB:
            stop_reason = "target_cost_reached"
            break

    correct_count = 0
    results_list = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        raw_answer = sample["answer"]
        match = re.search(r"####\s*([0-9,\.]+)", raw_answer)
        gold_answer = match.group(1).replace(",", "") if match else raw_answer

        final_pred = majority_vote(per_sample_preds[i])
        is_ok = (final_pred == gold_answer)
        if is_ok:
            correct_count += 1

        results_list.append({
            "sample_id": i + 1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": final_pred,
            "status": "CORRECT" if is_ok else "INCORRECT",
            "num_samples_voted": len(per_sample_preds[i]),
            "all_predictions": per_sample_preds[i],
            "all_replies": per_sample_replies[i],
        })

    accuracy = correct_count / n_samples * 100
    total_cost_rmb = compute_cost_rmb(
        total_usage["prompt_tokens"], total_usage["completion_tokens"]
    )

    payload = {
        "total_evaluated": n_samples,
        "target_cost_rmb": TARGET_COST_RMB,
        "passes_completed": passes_completed,
        "stop_reason": stop_reason,
        "correct_count": correct_count,
        "accuracy_percent": accuracy,
        "usage": {
            **total_usage,
            "cost_rmb": round(total_cost_rmb, 6),
        },
        "results": results_list,
    }

    output_path = "baseline_cost_matched_30.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print("\n========================================")
    print("     BASELINE (COST-MATCHED) SUMMARY    ")
    print("========================================")
    print(f"Total Evaluated: {n_samples}")
    print(f"Target Cost (RMB): {TARGET_COST_RMB:.4f}")
    print(f"Passes Completed: {passes_completed} ({stop_reason})")
    print(f"Accuracy: {correct_count}/{n_samples} ({accuracy:.1f}%)")
    print(f"API Calls: {total_usage['calls']}")
    print(f"Prompt Tokens: {total_usage['prompt_tokens']:,}")
    print(f"Completion Tokens: {total_usage['completion_tokens']:,}")
    print(f"Total Tokens: {total_usage['total_tokens']:,}")
    print(f"Total Cost (RMB): {total_cost_rmb:.4f}")
    print(f"Results Logged to: {output_path}")
    print("========================================")


if __name__ == "__main__":
    evaluate_baseline_cost_matched()
