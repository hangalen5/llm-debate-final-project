import json
import os
import re
from pathlib import Path
from openai import OpenAI

INPUT_JSON = Path("/Users/linndsey/llm-debate-final-project/evaluation_results_30.json")
OUTPUT_JSON = Path("/Users/linndsey/llm-debate-final-project/exp1_judge_bias_results.json")

MODEL = os.getenv("JUDGE_MODEL", "deepseek-chat")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("BASE_URL") or "https://api.deepseek.com"

INPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_INPUT_RMB_PER_1M", "2.0"))
OUTPUT_PRICE_RMB_PER_1M = float(os.getenv("DS_OUTPUT_RMB_PER_1M", "8.0"))


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def parse_json_decision(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            candidate = brace.group(0)
    try:
        data = json.loads(candidate)
        winner = str(data.get("winner", "")).strip().upper()
        if winner in {"A", "B", "TIE"}:
            return {"winner": winner, "reasoning": str(data.get("reasoning", "")).strip()}
    except Exception:
        pass

    upper = text.upper()
    if '"WINNER": "A"' in upper or "'WINNER': 'A'" in upper or re.search(r"\bWINNER\s*[:=]\s*A\b", upper):
        w = "A"
    elif '"WINNER": "B"' in upper or "'WINNER': 'B'" in upper or re.search(r"\bWINNER\s*[:=]\s*B\b", upper):
        w = "B"
    else:
        w = "UNKNOWN"
    return {"winner": w, "reasoning": text[:500]}


def call_judge(client: OpenAI, prompt: str, stats: dict) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a strict, impartial mathematical debate judge. Always reply with valid JSON."},
            {"role": "user", "content": prompt},
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
    return parse_json_decision(content or "")


def run_swapped_order(client: OpenAI, item: dict, stats: dict) -> dict:
    prompt = f"""Evaluate two final responses to the math word problem below.
Note: Response 1 is from Debater B, and Response 2 is from Debater A.

Problem:
{item["question"]}

Response 1 (Debater B):
{item["debater_b_final"]}

Response 2 (Debater A):
{item["debater_a_final"]}

Select the winner ("A" for Response 2 / Debater A, or "B" for Response 1 / Debater B).
Reply ONLY in JSON: {{"reasoning": "...", "winner": "A" or "B"}}"""
    return call_judge(client, prompt, stats)


def run_debiased_single_pass(client: OpenAI, question: str, first_text: str, second_text: str, stats: dict) -> dict:
    prompt = f"""Evaluate two anonymized solutions (Candidate 1 and Candidate 2) for the math problem below.

Rubric (strictly in priority order):
1. Mathematical correctness & alignment with the question asked (ignore minor formatting/casing differences).
2. Soundness of arithmetic steps and edge-case handling.
3. Do NOT reward verbosity, extra caveats, or presentation order. If both are mathematically sound and equivalent, choose "TIE".

Problem:
{question}

Candidate 1:
{first_text}

Candidate 2:
{second_text}

Reply ONLY in JSON: {{"reasoning": "...", "winner": "1", "2", or "TIE"}}"""
    res = call_judge(client, prompt, stats)
    raw_w = res["winner"]
    if raw_w == "1":
        res["mapped_winner"] = "FIRST"
    elif raw_w == "2":
        res["mapped_winner"] = "SECOND"
    elif raw_w == "TIE":
        res["mapped_winner"] = "TIE"
    else:
        res["mapped_winner"] = raw_w
    return res


def run_debiased_balanced(client: OpenAI, item: dict, stats: dict) -> dict:
    q = item["question"]
    a_txt = item["debater_a_final"]
    b_txt = item["debater_b_final"]

    p1 = run_debiased_single_pass(client, q, a_txt, b_txt, stats)
    w1 = "A" if p1["mapped_winner"] == "FIRST" else ("B" if p1["mapped_winner"] == "SECOND" else "TIE")

    p2 = run_debiased_single_pass(client, q, b_txt, a_txt, stats)
    w2 = "B" if p2["mapped_winner"] == "FIRST" else ("A" if p2["mapped_winner"] == "SECOND" else "TIE")

    if w1 == w2:
        final_w = w1
    elif "TIE" in {w1, w2}:
        final_w = w2 if w1 == "TIE" else w1
    else:
        final_w = "TIE"

    return {
        "winner": final_w,
        "pass_ab_winner": w1,
        "pass_ba_winner": w2,
        "consistent_across_swap": (w1 == w2),
        "reasoning_ab": p1["reasoning"],
        "reasoning_ba": p2["reasoning"],
    }


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_JSON}")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    results = data.get("results", [])
    client = get_client()

    swap_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_rmb": 0.0}
    debias_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_rmb": 0.0}

    orig_a = sum(1 for r in results if r.get("judge_decision", {}).get("winner") == "A")
    orig_b = sum(1 for r in results if r.get("judge_decision", {}).get("winner") == "B")

    swap_a = swap_b = flips_vs_orig = 0
    deb_a = deb_b = deb_tie = swap_consistent_cnt = 0

    records = []
    for item in results:
        sid = item["sample_id"]
        orig_w = item.get("judge_decision", {}).get("winner", "UNKNOWN")

        swap_res = run_swapped_order(client, item, swap_stats)
        sw_w = swap_res["winner"]
        if sw_w == "A":
            swap_a += 1
        elif sw_w == "B":
            swap_b += 1
        if orig_w in {"A", "B"} and sw_w in {"A", "B"} and sw_w != orig_w:
            flips_vs_orig += 1

        deb_res = run_debiased_balanced(client, item, debias_stats)
        dw = deb_res["winner"]
        if dw == "A":
            deb_a += 1
        elif dw == "B":
            deb_b += 1
        else:
            deb_tie += 1
        if deb_res["consistent_across_swap"]:
            swap_consistent_cnt += 1

        records.append({
            "sample_id": sid,
            "orig_winner_ab": orig_w,
            "swapped_winner_ba": sw_w,
            "flipped_on_swap": (orig_w in {"A", "B"} and sw_w in {"A", "B"} and sw_w != orig_w),
            "debiased_winner": dw,
            "debiased_pass_ab": deb_res["pass_ab_winner"],
            "debiased_pass_ba": deb_res["pass_ba_winner"],
            "debiased_consistent": deb_res["consistent_across_swap"],
        })
        print(f"[Sample {sid:02d}] Orig(A->B): {orig_w} | Swap(B->A): {sw_w} | Debiased: {dw} (AB={deb_res['pass_ab_winner']}, BA={deb_res['pass_ba_winner']})")

    n = len(results)
    summary = {
        "n_samples": n,
        "original_ab": {"A": orig_a, "B": orig_b, "A_pct": round(orig_a / n * 100, 1), "B_pct": round(orig_b / n * 100, 1)},
        "swapped_ba": {
            "A": swap_a,
            "B": swap_b,
            "A_pct": round(swap_a / n * 100, 1),
            "B_pct": round(swap_b / n * 100, 1),
            "flip_rate_vs_orig_pct": round(flips_vs_orig / n * 100, 1),
            "flips_count": flips_vs_orig,
            "stats": swap_stats,
        },
        "debiased_balanced": {
            "A": deb_a,
            "B": deb_b,
            "TIE": deb_tie,
            "A_pct": round(deb_a / n * 100, 1),
            "B_pct": round(deb_b / n * 100, 1),
            "TIE_pct": round(deb_tie / n * 100, 1),
            "swap_consistency_pct": round(swap_consistent_cnt / n * 100, 1),
            "stats": debias_stats,
        },
        "total_exp1_tokens": swap_stats["total_tokens"] + debias_stats["total_tokens"],
        "total_exp1_cost_rmb": round(swap_stats["cost_rmb"] + debias_stats["cost_rmb"], 4),
    }

    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== EXPERIMENT 1 SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
