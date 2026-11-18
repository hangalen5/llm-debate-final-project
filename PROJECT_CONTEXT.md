# Project Context: LLM Debate & Judge

## 1. Overview
- **Objective:** Build a Multi-agent Debate and LLM Judge system using the DeepSeek API (`deepseek-flash`), and compare it with same-cost baselines[cite: 2, 5].
- **Team Members:** [Your Name] & [Teammate's Name]

## 2. Current Status (Updated Nov 18, 2026)
- [x] `src/chat.py` calls `deepseek-flash` and prints API token usage.
- [x] GSM8K loader, direct baseline, self-consistency, and majority-vote debate work.
- [x] B ran a live 3-question smoke test; all three methods answered all three correctly (terminal screenshot, not saved experiment logs).
- [x] Local branch `codex/evaluation-results`: numeric answer extraction, automatic scoring, per-question CSV, accuracy summary, and run metadata implemented.
- [x] 38 offline evaluation tests passed, including negative/decimal answers and the full pipeline with mocked API responses. No paid API calls used for these tests.
- [ ] Review and merge the evaluation branch; run the updated pilot to produce real result files.
- [ ] Persistent per-call API logs, cost calculation, and caching still need implementation by A. Printed token usage is not a saved cost log.
- [ ] Confirm the research question, implement our own method (and judge if needed), match costs, and run held-out experiments.

### Evaluation handoff
- See `EVALUATION.md` for commands and output fields.
- Strict answer extraction accepts `Answer: <number>`, GSM8K `#### <number>`, or a bare number. It does not guess from unfinished reasoning. Fractions/percentages/equations are flagged as invalid rather than partially parsed.
- Results are saved under a new `results/pilot_<timestamp>/` directory for every run. Invalid predictions count as incorrect and are reported separately.
- This pilot still uses the first questions of the selected split; it is a smoke test, not a randomized or cost-matched experiment. Reserve separate questions for final evaluation.

## 3. Work Division
- Member A: `src/chat.py` (call logs, costs, cache), Debate Engine (`debate_engine.py`), and related tests.
- Member B: Dataset Loader (`dataset_loader.py`), evaluation metrics, and plots.

## 4. Team Workflow Rules
- Before coding: Run `git pull` and read this `PROJECT_CONTEXT.md` file.
- All major code changes must go through a Pull Request (PR) for peer review.
