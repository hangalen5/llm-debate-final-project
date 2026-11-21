import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

ROOT_DIR = Path("/Users/linndsey/llm-debate-final-project")
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from debate_engine import run_debate_with_judge  # noqa: E402

DEFAULT_OUTPUT = ROOT_DIR / "pilot_results.json"

MODEL = os.getenv("DEBATE_MODEL", "deepseek-chat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL)
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("BASE_URL") or "https://api.deepseek.com"

INPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_INPUT_RMB_PER_1M", "2.0"))
OUTPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_OUTPUT_RMB_PER_1M", "8.0"))


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def canonicalize_number(raw: str) -> str:
    text = (raw or "").strip().replace("$", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    num_str = match.group(0)
    try:
        val = float(num_str)
        return str(int(val)) if val.is_integer() else f"{val:g}"
    except ValueError:
        return num_str


def extract_gold_answer(gsm8k_answer: str) -> str:
    if "####" in gsm8k_answer:
        tail = gsm8k_answer.split("####")[-1].strip()
        return canonicalize_number(tail)
    return canonicalize_number(gsm8k_answer)


def extract_pred_answer(text: str) -> str:
    if not text:
        return ""
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        for bm in reversed(boxed_matches):
            cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", bm)
            ans = canonicalize_number(cleaned)
            if ans:
                return ans

    ans_lines = re.findall(r"(?:final answer|answer)\s*[:\-–]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if ans_lines:
        for line in reversed(ans_lines):
            ans = canonicalize_number(line)
            if ans:
                return ans

    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return canonicalize_number(nums[-1])
    return ""


def invoke_debate_engine(client: OpenAI, question: str, gold_answer: str) -> dict[str, Any]:
    sig = inspect.signature(run_debate_with_judge)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "client" in params:
        kwargs["client"] = client
    if "question" in params:
        kwargs["question"] = question
    elif "problem" in params:
        kwargs["problem"] = question
    if "gold_answer" in params:
        kwargs["gold_answer"] = gold_answer
    elif "answer" in params:
        kwargs["answer"] = gold_answer
    if "model" in params:
        kwargs["model"] = MODEL
    if "judge_model" in params:
        kwargs["judge_model"] = JUDGE_MODEL
    if "base_url" in params:
        kwargs["base_url"] = BASE_URL

    if kwargs:
        result = run_debate_with_judge(**kwargs)
    else:
        result = run_debate_with_judge(question)

    if isinstance(result, dict):
        return result
    return {"raw_output": result}


def normalize_record(sample_id: int, question: str, gold_answer: str, engine_out: dict[str, Any]) -> dict[str, Any]:
    a_final = str(
        engine_out.get("debater_a_final")
        or engine_out.get("final_a")
        or engine_out.get("a_final")
        or ""
    )
    b_final = str(
        engine_out.get("debater_b_final")
        or engine_out.get("final_b")
        or engine_out.get("b_final")
        or ""
    )

    judge_raw = engine_out.get("judge_decision") or engine_out.get("judge") or {}
    if isinstance(judge_raw, dict):
        winner = str(judge_raw.get("winner") or engine_out.get("winner") or "UNKNOWN").strip().upper()
        reasoning = str(judge_raw.get("reasoning") or engine_out.get("judge_reasoning") or "").strip()
        judge_decision = {"winner": winner, "reasoning": reasoning}
    else:
        winner = str(engine_out.get("winner") or "UNKNOWN").strip().upper()
        judge_decision = {"winner": winner, "reasoning": str(judge_raw).strip()}

    pred_a = canonicalize_number(str(engine_out.get("pred_a") or "")) or extract_pred_answer(a_final)
    pred_b = canonicalize_number(str(engine_out.get("pred_b") or "")) or extract_pred_answer(b_final)

    chosen_pred = canonicalize_number(str(engine_out.get("chosen_prediction") or engine_out.get("prediction") or ""))
    if not chosen_pred:
        if winner == "B" and pred_b:
            chosen_pred = pred_b
        elif pred_a:
            chosen_pred = pred_a
        else:
            chosen_pred = pred_b

    if "is_correct" in engine_out and isinstance(engine_out["is_correct"], bool):
        is_correct = engine_out["is_correct"]
    else:
        is_correct = chosen_pred == gold_answer

    return {
        "sample_id": sample_id,
        "question": question,
        "gold_answer": gold_answer,
        "debater_a_round1": engine_out.get("debater_a_round1", ""),
        "debater_b_round1": engine_out.get("debater_b_round1", ""),
        "debater_a_final": a_final,
        "debater_b_final": b_final,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "chosen_prediction": chosen_pred,
        "judge_decision": judge_decision,
        "is_correct": is_correct,
        "status": "CORRECT" if is_correct else "INCORRECT",
        "usage": engine_out.get("usage") or engine_out.get("token_usage") or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot GSM8K evaluation using src/debate_engine.py.")
    parser.add_argument("--n", type=int, default=int(os.getenv("PILOT_N", "10")), help="Number of pilot samples")
    parser.add_argument("--offset", type=int, default=int(os.getenv("PILOT_OFFSET", "0")), help="Dataset start offset")
    parser.add_argument("--split", type=str, default="test", help="GSM8K split")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON file path")
    args = parser.parse_args()

    client = get_client()
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    start_idx = max(0, args.offset)
    end_idx = min(len(ds), start_idx + max(0, args.n))
    subset = ds.select(range(start_idx, end_idx))

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost_rmb = 0.0
    total_calls = 0
    correct_count = 0
    records = []

    for idx, row in enumerate(subset, start=1):
        question = row["question"]
        gold = extract_gold_answer(row["answer"])
        engine_out = invoke_debate_engine(client, question, gold)
        rec = normalize_record(idx, question, gold, engine_out)
        records.append(rec)

        if rec["is_correct"]:
            correct_count += 1

        u = rec.get("usage") or {}
        in_tok = int(u.get("prompt_tokens", 0) or 0)
        out_tok = int(u.get("completion_tokens", 0) or 0)
        tok = int(u.get("total_tokens", 0) or (in_tok + out_tok))
        calls = int(engine_out.get("api_calls") or engine_out.get("calls") or 0)
        cost = engine_out.get("cost_rmb")
        if cost is None and (in_tok or out_tok or tok):
            eff_in = in_tok if (in_tok or out_tok) else int(tok * 0.8)
            eff_out = out_tok if (in_tok or out_tok) else int(tok * 0.2)
            cost = (eff_in / 1_000_000.0) * INPUT_PRICE_RMB_PER_1M + (
                eff_out / 1_000_000.0
            ) * OUTPUT_PRICE_RMB_PER_1M

        total_prompt_tokens += in_tok
        total_completion_tokens += out_tok
        total_tokens += tok
        total_calls += calls
        if cost is not None:
            total_cost_rmb += float(cost)

        print(
            f"[Pilot {idx:02d}/{len(subset):02d}] Gold={gold} | Pred={rec['chosen_prediction']} "
            f"| Winner={rec['judge_decision']['winner']} | {rec['status']}"
        )

    n_eval = len(records)
    summary = {
        "mode": "pilot",
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "split": args.split,
        "offset": start_idx,
        "n_samples": n_eval,
        "correct_count": correct_count,
        "accuracy_pct": round(correct_count / n_eval * 100, 1) if n_eval else 0.0,
        "total_calls": total_calls,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_rmb": round(total_cost_rmb, 4),
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": records}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved pilot results to {output_path}")


if __name__ == "__main__":
    main()