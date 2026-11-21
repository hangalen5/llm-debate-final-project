import json
import os
import re
from collections import Counter
from pathlib import Path
from openai import OpenAI

INPUT_JSON = Path("/Users/linndsey/llm-debate-final-project/evaluation_results_30.json")
OUTPUT_JSON = Path("/Users/linndsey/llm-debate-final-project/exp2_self_consistency_results.json")

MODEL = os.getenv("DEBATE_MODEL", "deepseek-chat")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("BASE_URL") or "https://api.deepseek.com"

SC_K = int(os.getenv("SC_K", "5"))
TEMPERATURE = float(os.getenv("SC_TEMPERATURE", "0.7"))
MAX_TOTAL_TOKENS = int(os.getenv("SC_MAX_TOTAL_TOKENS", "0"))

INPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_INPUT_RMB_PER_1M", "2.0"))
OUTPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_OUTPUT_RMB_PER_1M", "8.0"))


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def canonicalize_number(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("$", "").replace(",", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return ""
    num_str = m.group(0)
    try:
        val = float(num_str)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    except ValueError:
        return num_str


def extract_answer(text: str) -> str:
    if not text:
        return ""
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        for bm in reversed(boxed_matches):
            cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", bm)
            num = canonicalize_number(cleaned)
            if num:
                return num

    ans_lines = re.findall(r"(?:final answer|answer)\s*[:\-–]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if ans_lines:
        for line in reversed(ans_lines):
            num = canonicalize_number(line)
            if num:
                return num

    all_nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if all_nums:
        return canonicalize_number(all_nums[-1])
    return ""


def majority_vote(preds: list[str]) -> str:
    valid = [p for p in preds if p != ""]
    if not valid:
        return ""
    counts = Counter(valid)
    top_count = max(counts.values())
    tied = [ans for ans, cnt in counts.items() if cnt == top_count]
    if len(tied) == 1:
        return tied[0]
    for p in reversed(valid):
        if p in tied:
            return p
    return tied[0]


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_JSON}")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    results = data.get("results", [])
    client = get_client()

    stats = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_rmb": 0.0,
        "stopped_early_by_token_cap": False,
    }

    sc_correct = 0
    first_pass_correct = 0
    records = []

    for item in results:
        sid = item["sample_id"]
        question = item["question"]
        gold = canonicalize_number(item["gold_answer"])

        sample_preds = []
        for _ in range(SC_K):
            if MAX_TOTAL_TOKENS > 0 and stats["total_tokens"] >= MAX_TOTAL_TOKENS:
                stats["stopped_early_by_token_cap"] = True
                break

            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=450,
                messages=[
                    {
                        "role": "system",
                        "content": "Solve the math word problem step by step clearly, and put the final numeric answer inside \\boxed{}.",
                    },
                    {"role": "user", "content": question},
                ],
            )
            usage = resp.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            stats["calls"] += 1
            stats["prompt_tokens"] += in_tok
            stats["completion_tokens"] += out_tok
            stats["total_tokens"] += in_tok + out_tok
            stats["cost_rmb"] += (in_tok / 1_000_000.0) * INPUT_PRICE_RMB_PER_1M + (
                out_tok / 1_000_000.0
            ) * OUTPUT_PRICE_RMB_PER_1M

            content = resp.choices[0].message.content if resp.choices else ""
            pred = extract_answer(content or "")
            sample_preds.append(pred)

        if not sample_preds:
            break

        if sample_preds[0] == gold:
            first_pass_correct += 1

        voted_pred = majority_vote(sample_preds)
        is_correct = (voted_pred == gold)
        if is_correct:
            sc_correct += 1

        records.append({
            "sample_id": sid,
            "gold_answer": gold,
            "sc_samples": sample_preds,
            "sc_voted_prediction": voted_pred,
            "sc_correct": is_correct,
            "single_pass_first_correct": (sample_preds[0] == gold),
        })
        print(
            f"[Sample {sid:02d}] Gold={gold} | SC@{len(sample_preds)}={voted_pred} "
            f"('{'CORRECT' if is_correct else 'INCORRECT'}') | draws={sample_preds}"
        )

    evaluated_n = len(records)
    summary = {
        "evaluated_samples": evaluated_n,
        "sc_k": SC_K,
        "temperature": TEMPERATURE,
        "sc_accuracy_pct": round(sc_correct / evaluated_n * 100, 1) if evaluated_n else 0.0,
        "sc_correct_count": sc_correct,
        "single_pass_1st_draw_accuracy_pct": round(first_pass_correct / evaluated_n * 100, 1) if evaluated_n else 0.0,
        "single_pass_1st_draw_correct_count": first_pass_correct,
        "total_tokens": stats["total_tokens"],
        "total_calls": stats["calls"],
        "estimated_cost_rmb": round(stats["cost_rmb"], 4),
        "stopped_early_by_token_cap": stats["stopped_early_by_token_cap"],
    }

    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== EXPERIMENT 2 SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
