import json
import re
from chat import chat

def run_debate_with_judge(question: str, n_rounds: int = 1) -> dict:
    """
    Standard LLM Debate Engine:
    - Round 0: Debater A and Debater B answer the question independently.
    - Round 1..N: Each debater reads the other's previous response, defends their stance, or updates it.
    - Judge: An impartial judge reads the final arguments and returns a structured JSON 
      containing reasoning and the winner ('A' or 'B').
    """
    # --- ROUND 0: Independent initial answers ---
    prompt_a_r0 = (
        f"You are Debater A. Answer the following math question step by step "
        f"and clearly state your final numerical answer.\n\nQuestion: {question}"
    )
    prompt_b_r0 = (
        f"You are Debater B. Answer the following math question step by step "
        f"and clearly state your final numerical answer.\n\nQuestion: {question}"
    )
    
    reply_a = chat(prompt_a_r0)
    reply_b = chat(prompt_b_r0)

    # --- ROUND 1..N: Critique and Defense ---
    for r in range(1, n_rounds + 1):
        prompt_a = (
            f"Question: {question}\n\n"
            f"Your previous answer (Debater A):\n{reply_a}\n\n"
            f"Debater B's answer:\n{reply_b}\n\n"
            f"Read Debater B's argument. Defend your answer or update it if B is right. "
            f"Point out any flaws in B's reasoning."
        )
        prompt_b = (
            f"Question: {question}\n\n"
            f"Your previous answer (Debater B):\n{reply_b}\n\n"
            f"Debater A's answer:\n{reply_a}\n\n"
            f"Read Debater A's argument. Defend your answer or update it if A is right. "
            f"Point out any flaws in A's reasoning."
        )
        
        reply_a = chat(prompt_a)
        reply_b = chat(prompt_b)

    # --- JUDGE: Evaluate final arguments and output JSON ---
    judge_prompt = (
        f"You are an impartial Judge. Read the question and the final arguments from Debater A and Debater B.\n"
        f"Determine which debater has the correct and more sound reasoning.\n\n"
        f"Question: {question}\n\n"
        f"Debater A's final reply:\n{reply_a}\n\n"
        f"Debater B's final reply:\n{reply_b}\n\n"
        f'Respond ONLY with a valid JSON object in this exact format: {{"reasoning": "<your explanation>", "winner": "A" or "B"}}'
    )
    
    judge_raw = chat(judge_prompt)
    
    # Extract JSON response safely from judge output
    try:
        json_match = re.search(r"\{.*\}", judge_raw, re.DOTALL)
        judge_result = json.loads(json_match.group(0)) if json_match else {"reasoning": judge_raw, "winner": "Unknown"}
    except Exception:
        judge_result = {"reasoning": judge_raw, "winner": "Error parsing JSON"}

    # Extract final numerical prediction from the winner's text
    winner_text = reply_a if judge_result.get("winner") == "A" else reply_b
    
    # Robust extraction handling LaTeX formatting like \boxed{...} with comma/braces
    final_pred = ""
    boxed_match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", winner_text)
    if boxed_match:
        boxed_content = boxed_match.group(1).replace("{,}", "").replace(",", "")
        nums_in_box = re.findall(r"[0-9]+(?:\.[0-9]+)?", boxed_content)
        if nums_in_box:
            final_pred = nums_in_box[-1]
            
    if not final_pred:
        # Fallback 1: look for bolded numbers like **70,000** or **$70,000**
        bold_match = re.search(r"\*\*\$?([0-9,\.]+)", winner_text)
        if bold_match:
            final_pred = bold_match.group(1).replace(",", "")
        else:
            # Fallback 2: grab the last standalone number in the text
            numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", winner_text)
            final_pred = numbers[-1].replace(",", "") if numbers else ""

    return {
        "debater_a_final": reply_a,
        "debater_b_final": reply_b,
        "judge": judge_result,
        "prediction": final_pred
    }