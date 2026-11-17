from collections import Counter
from chat import chat  # Import trực tiếp từ file chat.py nằm chung trong thư mục src/
from dataset_loader import extract_gsm8k_answer

def get_majority_vote(answers: list) -> str:
    valid = [a for a in answers if a]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]

# --- 1. BASELINE: DIRECT (1 CALL) ---
def run_direct_baseline(question: str) -> str:
    prompt = (
        f"Solve the following math problem step by step. "
        f"End your response with 'Answer: <number>'.\n\nQuestion: {question}"
    )
    response = chat(prompt)
    return extract_gsm8k_answer(response)

# --- 2. BASELINE: SELF-CONSISTENCY (k CALLS) ---
def run_self_consistency(question: str, k: int = 5) -> str:
    answers = []
    for i in range(k):
        prompt = (
            f"Attempt {i+1}: Solve the following math problem step by step. "
            f"End your response with 'Answer: <number>'.\n\nQuestion: {question}"
        )
        response = chat(prompt)
        answers.append(extract_gsm8k_answer(response))
    return get_majority_vote(answers)

# --- 3. DEBATE ENGINE (N AGENTS x R ROUNDS) ---
def run_debate(question: str, n_agents: int = 3, n_rounds: int = 2) -> str:
    # Round 0: Each agent answers independently
    agent_histories = []
    for i in range(n_agents):
        prompt = (
            f"You are Debater {i+1}. Solve this problem step by step. "
            f"End with 'Answer: <number>'.\n\nQuestion: {question}"
        )
        reply = chat(prompt)
        agent_histories.append(reply)

    # Round 1 to R: Read other debaters' responses and update
    for r in range(1, n_rounds + 1):
        new_histories = []
        for i in range(n_agents):
            others_text = "\n\n".join(
                [f"--- Debater {j+1}'s response ---\n{agent_histories[j]}" 
                 for j in range(n_agents) if j != i]
            )
            prompt = (
                f"Question: {question}\n\n"
                f"Your previous response:\n{agent_histories[i]}\n\n"
                f"Other debaters' responses:\n{others_text}\n\n"
                f"Carefully review their reasoning. Point out any flaws in their logic or your own, "
                f"then provide your updated solution. End with 'Answer: <number>'."
            )
            reply = chat(prompt)
            new_histories.append(reply)
        agent_histories = new_histories

    # Aggregate final answers using Majority Vote
    final_answers = [extract_gsm8k_answer(ans) for ans in agent_histories]
    return get_majority_vote(final_answers)