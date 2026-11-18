"""Run a small GSM8K pilot and save scores. This command makes paid API calls."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .answers import extract_gold_answer, is_correct, normalize_number
    from .dataset_loader import load_gsm8k_sample
    from .debate_engine import run_direct_baseline, run_self_consistency, run_debate
else:
    from answers import extract_gold_answer, is_correct, normalize_number
    from dataset_loader import load_gsm8k_sample
    from debate_engine import run_direct_baseline, run_self_consistency, run_debate


RESULT_FIELDS = [
    "question_id", "question", "method", "gold_answer", "prediction",
    "correct", "status",
]


def evaluate_pilot(n=3, split="test", output_dir=None):
    print("Loading GSM8K sample dataset...")
    samples = load_gsm8k_sample(n=n, split=split)
    if not samples:
        raise ValueError("No questions were loaded")
    # Validate all answer keys before spending any API budget.
    gold_answers = [extract_gold_answer(sample["answer"]) for sample in samples]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parents[1] / "results" / f"pilot_{run_id}"
    )
    # Refuse to overwrite previous experiment results.
    destination.mkdir(parents=True, exist_ok=False)
    methods = [
        ("direct", run_direct_baseline),
        ("self_consistency", lambda q: run_self_consistency(q, k=3)),
        ("debate", lambda q: run_debate(q, n_agents=3, n_rounds=1)),
    ]
    metadata = {
        "run_id": run_id, "dataset": "openai/gsm8k", "config": "main",
        "split": split, "sampling": "first_n", "requested_n": n,
        "loaded_n": len(samples), "self_consistency_k": 3,
        "debate_agents": 3, "debate_rounds": 1, "complete": False,
    }
    metadata_path = destination / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    totals = {name: {"correct": 0, "total": 0, "invalid": 0} for name, _ in methods}
    print(f"Loaded {len(samples)} samples. Results directory: {destination.resolve()}")
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        handle.flush()
        for index, (sample, gold) in enumerate(zip(samples, gold_answers)):
            question_id = sample.get("question_id", f"gsm8k:{split}:{index}")
            print(f"\n--- {question_id} ---\nQuestion: {sample['question']}\nGold Answer: {gold}")
            for method, run_method in methods:
                prediction = normalize_number(run_method(sample["question"]))
                correct = is_correct(prediction, gold)
                status = "ok" if prediction else "no_valid_prediction"
                writer.writerow({
                    "question_id": question_id, "question": sample["question"],
                    "method": method, "gold_answer": gold, "prediction": prediction,
                    "correct": correct, "status": status,
                })
                # Preserve completed rows even if a later call is interrupted.
                handle.flush()
                totals[method]["total"] += 1
                totals[method]["correct"] += int(correct)
                totals[method]["invalid"] += int(not prediction)
                print(f"-> {method}: {prediction or '(no valid answer)'} | correct={correct}")
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "correct", "total", "invalid", "accuracy"])
        writer.writeheader()
        print("\nSummary (invalid answers count as incorrect):")
        for method, counts in totals.items():
            accuracy = counts["correct"] / counts["total"]
            writer.writerow({"method": method, **counts, "accuracy": accuracy})
            print(f"{method}: {counts['correct']}/{counts['total']} = {accuracy:.1%}")
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions.csv, summary.csv and run.json to {destination.resolve()}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, help="A new directory; existing directories are never overwritten")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    evaluate_pilot(n=args.n, split=args.split, output_dir=args.output_dir)
