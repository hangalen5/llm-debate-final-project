import csv
import json
import re
from pathlib import Path
from typing import Optional

INPUT_PRICE_PER_1M = 2.0
OUTPUT_PRICE_PER_1M = 3.0

BASE_ROWS = [
    {
        "run_id": "debate_eval_30",
        "experiment": "Debate Evaluation (N=30)",
        "scope": "N=30",
        "metric": "24/30 (80.0%)",
        "total_tokens": 0,
        "cost_rmb": "Check run log",
    },
    {
        "run_id": "exp1a_judge_swapped_ba",
        "experiment": "Judge Position Bias (Swapped B/A)",
        "scope": "N=30 (30 calls)",
        "metric": "Flip=43.3%",
        "total_tokens": 19048,
        "cost_rmb": "0.0624",
    },
    {
        "run_id": "exp1b_judge_debiased_2pass",
        "experiment": "Judge Debiasing (2-Pass)",
        "scope": "N=30 (60 calls)",
        "metric": "TIE=100.0%",
        "total_tokens": 36590,
        "cost_rmb": "0.1041",
    },
    {
        "run_id": "exp1_total",
        "experiment": "Experiment 1 Total",
        "scope": "N=30 (90 calls)",
        "metric": "Cons=100.0%",
        "total_tokens": 55638,
        "cost_rmb": "0.1664",
    },
    {
        "run_id": "exp2_sc_at_20",
        "experiment": "Self-Consistency (k=20)",
        "scope": "N=30 (600 calls)",
        "metric": "100.0% (30/30)",
        "total_tokens": 172926,
        "cost_rmb": "1.0816",
    },
]


def find_debate_log() -> Optional[Path]:
    candidates = []
    for ext in ("*.jsonl", "*.json", "*.log", "*.txt"):
        for p in Path(".").rglob(ext):
            name_low = p.name.lower()
            if "debate_eval_30" in name_low or (
                "debate" in name_low
                and ("log" in name_low or "eval" in name_low or "result" in name_low)
            ):
                candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def parse_usage_from_log(log_path: Path) -> dict:
    total_in = 0
    total_out = 0
    total_tokens = 0
    calls = 0

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            usage = data.get("usage") if isinstance(data, dict) else None
            src = usage if isinstance(usage, dict) else (data if isinstance(data, dict) else {})

            in_tok = int(src.get("prompt_tokens") or src.get("input_tokens") or 0)
            out_tok = int(src.get("completion_tokens") or src.get("output_tokens") or 0)
            tot_tok = int(src.get("total_tokens") or (in_tok + out_tok))

            if tot_tok > 0:
                total_in += in_tok
                total_out += out_tok
                total_tokens += tot_tok
                calls += 1
            continue
        except json.JSONDecodeError:
            pass

        m_tot = re.search(r"total_tokens\D+(\d+)", line, re.IGNORECASE)
        if m_tot:
            total_tokens += int(m_tot.group(1))
            calls += 1

    cost_rmb = (total_in / 1_000_000) * INPUT_PRICE_PER_1M + (
        total_out / 1_000_000
    ) * OUTPUT_PRICE_PER_1M

    return {
        "calls": calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_tokens,
        "cost_rmb": f"{cost_rmb:.4f}" if total_tokens > 0 else "Check run log",
    }


def print_markdown_table(rows: list) -> None:
    print("\n| Experiment / Run | Scope | Metric | Total Tokens | Cost (RMB) |")
    print("| :--- | :--- | :--- | ---: | :--- |")
    for r in rows:
        tok_str = (
            f"{r['total_tokens']:,}"
            if isinstance(r["total_tokens"], int)
            else str(r["total_tokens"])
        )
        print(f"| {r['experiment']} | {r['scope']} | {r['metric']} | {tok_str} | {r['cost_rmb']} |")


def save_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "scope", "metric", "total_tokens", "cost_rmb"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})


def main() -> None:
    log_file = find_debate_log()
    rows = [dict(r) for r in BASE_ROWS]

    if log_file:
        print(f"Found log file: {log_file}")
        stats = parse_usage_from_log(log_file)
        print(f"Log stats: {stats}")
        if stats["total_tokens"] > 0:
            rows[0]["total_tokens"] = stats["total_tokens"]
            rows[0]["cost_rmb"] = stats["cost_rmb"]
            if stats["calls"] > 0:
                rows[0]["scope"] = f"N=30 ({stats['calls']} calls)"
    else:
        print("No log file found for debate_eval_30; keeping 0 and 'Check run log'.")

    print_markdown_table(rows)
    csv_out = Path("results/summary_table.csv")
    save_csv(rows, csv_out)
    print(f"\nSaved CSV summary to: {csv_out}")


if __name__ == "__main__":
    main()
