"""Offline regression tests. No API key, network, or paid model calls needed."""

import csv
import json
from unittest.mock import Mock

import pytest

from src.answers import extract_gsm8k_answer, extract_gold_answer, is_correct
from src import debate_engine, run_pilot


@pytest.mark.parametrize("response,expected", [
    ("Calculation: 2 - 5.\nAnswer: -3", "-3"),
    ("Answer: -0.75", "-0.75"),
    ("Answer: .50", "0.5"),
    ("Answer: 70,000", "70000"),
    ("Answer: -1,234.50", "-1234.5"),
    ("Answer: 3.000", "3"),
    ("Answer: −3", "-3"),
    ("Answer: -0.0", "0"),
    ("Answer: $18.00", "18"),
    ("Answer: 18 dollars.", "18"),
    ("Answer: 18.", "18"),
    ("ANSWER: +3", "3"),
    ("Reasoning\n#### -3.50", "-3.5"),
    ("Answer: 8\nCorrection\nAnswer: 9", "9"),
    ("3.0", "3"),
    ("We have 16 eggs and eat 3, but I cannot finish.", ""),
    ("Answer: 1/2", ""),
    ("Answer: 2 + 3", ""),
    ("Answer: 1,23", ""),
    ("Answer: 3.4.5", ""),
    ("Answer: 30%", ""),
    ("Answer: NaN", ""),
    ("Answer: 18\nAnswer: unknown", ""),
    ("Answer: 18\nAnswer:\nI cannot decide.", ""),
    ("", ""),
])
def test_extract_answer(response, expected):
    assert extract_gsm8k_answer(response) == expected


@pytest.mark.parametrize("prediction,gold,expected", [
    ("3.00", "3", True), ("70,000", "70000", True),
    ("-0.75", "-.750", True), ("-3", "3", False),
    ("", "0", False), ("no answer", "18", False),
    ("0.30000000000000004", "0.3", False),
])
def test_score(prediction, gold, expected):
    assert is_correct(prediction, gold) is expected


def test_invalid_gold_is_not_silently_scored():
    assert extract_gold_answer("Some reasoning\n#### -1,200.50") == "-1200.5"
    for raw in ("reasoning contains 18", "#### unknown", "#### 1/2"):
        with pytest.raises(ValueError):
            extract_gold_answer(raw)
    with pytest.raises(ValueError):
        is_correct("", "")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def offline_methods(monkeypatch):
    samples = [
        {"question_id": "gsm8k:test:0", "question": 'A question, with "quotes"\nand a newline', "answer": "Steps\n#### -3"},
        {"question_id": "gsm8k:test:1", "question": "Second question", "answer": "Steps\n#### 0.5"},
    ]
    loader = Mock(return_value=samples)
    monkeypatch.setattr(run_pilot, "load_gsm8k_sample", loader)
    monkeypatch.setattr(run_pilot, "run_direct_baseline", Mock(side_effect=["-3.0", ""]))
    monkeypatch.setattr(run_pilot, "run_self_consistency", Mock(side_effect=["3", ".50"]))
    monkeypatch.setattr(run_pilot, "run_debate", Mock(side_effect=["-3", "0.5"]))
    return samples, loader


def test_saved_scores_and_summary(tmp_path, offline_methods):
    samples, loader = offline_methods
    out = run_pilot.evaluate_pilot(n=2, output_dir=tmp_path / "run")
    loader.assert_called_once_with(n=2, split="test")
    rows = read_rows(out / "predictions.csv")
    assert len(rows) == 6
    assert rows[0]["question"] == samples[0]["question"]
    assert rows[0]["question_id"] == "gsm8k:test:0"
    assert [r["correct"] for r in rows] == ["True", "False", "True", "False", "True", "True"]
    assert rows[3]["status"] == "no_valid_prediction"
    summary = {row["method"]: row for row in read_rows(out / "summary.csv")}
    assert summary["direct"]["accuracy"] == "0.5"
    assert summary["direct"]["total"] == "2"
    assert summary["direct"]["invalid"] == "1"
    assert summary["self_consistency"]["accuracy"] == "0.5"
    assert summary["debate"]["accuracy"] == "1.0"
    assert json.loads((out / "run.json").read_text())["complete"] is True
    before = (out / "predictions.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_pilot.evaluate_pilot(n=2, output_dir=out)
    assert (out / "predictions.csv").read_bytes() == before


def test_bad_key_stops_before_api_calls(tmp_path, monkeypatch, offline_methods):
    monkeypatch.setattr(run_pilot, "load_gsm8k_sample", Mock(return_value=[
        {"question": "bad key", "answer": "Answer missing"},
    ]))
    with pytest.raises(ValueError):
        run_pilot.evaluate_pilot(output_dir=tmp_path / "bad")
    run_pilot.run_direct_baseline.assert_not_called()
    assert not (tmp_path / "bad").exists()


def test_interrupted_run_keeps_finished_rows(tmp_path, monkeypatch, offline_methods):
    monkeypatch.setattr(run_pilot, "run_self_consistency", Mock(side_effect=RuntimeError("interrupted")))
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_pilot.evaluate_pilot(n=2, output_dir=out)
    assert len(read_rows(out / "predictions.csv")) == 1
    assert json.loads((out / "run.json").read_text())["complete"] is False
    assert not (out / "summary.csv").exists()


def test_real_pipeline_with_mocked_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "load_gsm8k_sample", Mock(return_value=[
        {"question_id": "gsm8k:test:0", "question": "2 - 5 = ?", "answer": "2 - 5 = -3\n#### -3"},
    ]))
    chat = Mock(return_value="Reasoning\nAnswer: -3.00")
    monkeypatch.setattr(debate_engine, "chat", chat)
    out = run_pilot.evaluate_pilot(n=1, output_dir=tmp_path / "integrated")
    assert chat.call_count == 10  # direct 1 + self-consistency 3 + debate 6
    assert all(row["correct"] == "True" for row in read_rows(out / "predictions.csv"))
    for call in chat.call_args_list:
        assert "####" not in call.args[0]  # Never send the answer key to the model.


def test_majority_vote_uses_normalized_answers(monkeypatch):
    monkeypatch.setattr(debate_engine, "chat", Mock(side_effect=["Answer: 3", "Answer: 3.0", "Answer: 4"]))
    assert debate_engine.run_self_consistency("Question", k=3) == "3"
