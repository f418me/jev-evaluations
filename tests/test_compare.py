import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_fomc import CASES

from compare import (
    ComparisonConfig,
    ModelSpec,
    evaluate,
    markdown_report,
    run_comparison,
)
from fomc import ROOT, Baseline
from llm import build_llm_agent


@pytest.fixture
def baseline():
    return Baseline.model_validate_json((ROOT / "data/baseline.json").read_text())


def config():
    return ComparisonConfig(
        models=[
            ModelSpec(name="Jev", kind="jev", model="typesafe:jev-latest"),
            ModelSpec(name="LLM A", kind="llm", model="openai:test-a"),
            ModelSpec(name="LLM B", kind="llm", model="openai:test-b"),
        ]
    )


@pytest.mark.parametrize(
    "output_type,args,expected",
    [
        (
            "FEDDecisionImpact",
            {
                "impact_on_bitcoin": "positive",
                "confidence": 0.91,
                "reasoning": "Hold instead of hike",
                "actual_fed_decision_summary": "Hold",
                "actual_interest_rate_change_type": "hold",
                "actual_interest_rate_change_amount": "0bp",
                "actual_narrative": "dovish",
            },
            "positive",
        ),
        ("IrrelevantFEDContent", {"reason": "Preview only"}, "irrelevant"),
        ("FailedFEDAnalysis", {"error_message": "Insufficient information"}, "failed"),
    ],
)
def test_llm_union_and_prompt(baseline, output_type, args, expected):
    from fomc import Statement

    def respond(messages, info):
        system = messages[0].parts[0].content
        assert baseline.notes in system
        assert "has conflicting signals" in system
        assert "rate surprise takes priority" not in system
        assert messages[0].parts[1].content == "Statement content only"
        tool = next(t for t in info.output_tools if t.name.endswith(output_type))
        return ModelResponse(
            parts=[ToolCallPart(tool.name, args)], model_name="mock-llm"
        )

    agent = build_llm_agent(FunctionModel(respond), baseline)
    report = evaluate(
        agent,
        config().models[1],
        baseline,
        Statement(content="Statement content only"),
        0.99,
    )
    assert report["outcome"] == expected
    assert report["elapsed_seconds"] >= 0
    assert report["raw_output"] == args
    if expected == "positive":
        # Jev's threshold must not be applied to the old LLM self-assessment.
        assert report["confidence"] == 0.91


def test_batch_order_failures_timing_and_markdown(baseline, monkeypatch):
    import compare

    def factory(spec, baseline):
        if spec.name == "LLM B":
            raise ValueError("missing credential; sensitive detail must not appear")
        return object()

    def evaluate_stub(agent, spec, baseline, statement, threshold):
        if spec.name == "LLM A":
            raise TimeoutError("secret detail")
        return {
            "status": "assessed",
            "outcome": "positive",
            "elapsed_seconds": 0.75,
            "confidence": 1.0,
            "reasoning": "Rate surprise",
            "resolved_model": "jev-test",
        }

    monkeypatch.setattr(compare, "evaluate", evaluate_stub)
    case = next(c for c in CASES if c["name"] == "surprise-hold")
    report = run_comparison(
        config(), baseline, [case], repeats=2, agent_factory=factory
    )
    assert len(report["results"]) == 6
    assert [r["name"] for r in report["results"]] == [
        "Jev",
        "LLM A",
        "LLM B",
        "LLM A",
        "LLM B",
        "Jev",
    ]
    assert report["summary"][0]["passed"] == 2
    assert report["summary"][0]["median_seconds"] == 0.75
    assert report["summary"][1]["errors"] == 2
    assert report["summary"][1]["mean_seconds"] is None
    assert report["results"][1]["elapsed_seconds"] >= 0
    assert report["results"][2]["elapsed_seconds"] is None
    assert "secret" not in json.dumps(report)
    text = markdown_report(report)
    assert "0.750" in text and "positive" in text and "TimeoutError" in text


def test_preview_uses_explicit_llm_expectation(baseline, monkeypatch):
    import compare

    def stub(agent, spec, *args):
        outcome = "abstain" if spec.kind == "jev" else "irrelevant"
        return {"status": outcome, "outcome": outcome, "elapsed_seconds": 0.5}

    monkeypatch.setattr(compare, "evaluate", stub)
    case = next(c for c in CASES if c["name"] == "preview")
    report = run_comparison(
        config(), baseline, [case], agent_factory=lambda *args: object()
    )
    assert all(r["passed"] for r in report["results"])


def test_config_rejects_duplicates_and_wrong_provider():
    with pytest.raises(ValidationError):
        ComparisonConfig(models=[config().models[0], config().models[0]])
    with pytest.raises(ValidationError):
        ModelSpec(name="bad", kind="llm", model="typesafe:jev-latest")


def test_comparison_cli_writes_reports(tmp_path, monkeypatch):
    import compare

    report = {"results": [], "summary": []}
    monkeypatch.setattr(compare, "run_comparison", lambda *args: report)
    target = tmp_path / "comparison.json"
    monkeypatch.setattr("sys.argv", ["compare.py", "--output", str(target)])
    assert compare.main() == 0
    assert json.loads(target.read_text()) == report
    assert "Individual decisions" in target.with_suffix(".md").read_text()


@pytest.mark.parametrize(
    "selection,expected_cases",
    [
        ([], [case["name"] for case in CASES]),
        (["--case", "surprise-hold", "dovish-hike"], ["surprise-hold", "dovish-hike"]),
    ],
)
def test_jev_only_cli_runs_selected_cases(
    tmp_path, monkeypatch, selection, expected_cases
):
    import compare

    # Keep actual batch execution; only the provider boundary is mocked.
    config_path = tmp_path / "models.json"
    config_path.write_text(config().model_dump_json())
    initialized = []

    def factory(spec, baseline):
        initialized.append(spec.kind)
        assert spec.kind == "jev"
        return object()

    def batch(config, baseline, cases, repeats, threshold):
        return run_comparison(config, baseline, cases, repeats, threshold, factory)

    monkeypatch.setattr(compare, "run_comparison", batch)
    monkeypatch.setattr(
        compare,
        "evaluate",
        lambda *args: {
            "status": "assessed",
            "outcome": "positive",
            "elapsed_seconds": 0.5,
        },
    )
    target = tmp_path / "jev.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare.py",
            "--jev-only",
            "--config",
            str(config_path),
            "--output",
            str(target),
            "--repeat",
            "2",
            *selection,
        ],
    )
    exit_code = compare.main()
    report = json.loads(target.read_text())
    assert initialized == ["jev"]
    assert [row["case"] for row in report["results"]] == expected_cases * 2
    assert exit_code == int(any(not row["passed"] for row in report["results"]))
    assert report["summary"][0]["runs"] == len(expected_cases) * 2
    assert "# FOMC: Jev case tests" in target.with_suffix(".md").read_text()


@pytest.mark.parametrize("missing_jev", [False, True])
def test_cli_rejects_invalid_selection_before_running(
    tmp_path, monkeypatch, missing_jev
):
    import compare

    selected = config()
    if missing_jev:
        selected = ComparisonConfig(models=selected.models[1:])
    path = tmp_path / "models.json"
    path.write_text(selected.model_dump_json())
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare.py",
            "--config",
            str(path),
            "--jev-only",
            "--case",
            "surprise-hold",
            "typo",
        ],
    )
    monkeypatch.setattr(
        compare, "run_comparison", lambda *args: pytest.fail("Must not run")
    )
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert exc.value.code == 2


def test_comparison_cli_serializes_nested_provider_datetime(tmp_path, monkeypatch):
    import compare

    timestamp = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    report = {
        "results": [
            {
                "case": "surprise-hold",
                "repeat": 1,
                "name": "LLM",
                "requested_model": "openai:test",
                "resolved_model": "test",
                "outcome": "positive",
                "expected": "positive",
                "passed": True,
                "elapsed_seconds": 0.75,
                "confidence": 0.95,
                "reasoning": "Hold instead of hike",
                "provider_details": {
                    "created_at": timestamp,
                    "nested": [{"timestamp": timestamp}],
                },
            }
        ],
        "summary": [],
    }
    monkeypatch.setattr(compare, "run_comparison", lambda *args: report)
    target = tmp_path / "comparison.json"
    monkeypatch.setattr("sys.argv", ["compare.py", "--output", str(target)])
    assert compare.main() == 0
    saved = json.loads(target.read_text())["results"][0]
    assert saved["provider_details"]["created_at"] == "2026-09-24T12:30:00Z"
    assert saved["provider_details"]["nested"][0]["timestamp"] == "2026-09-24T12:30:00Z"
    assert saved["elapsed_seconds"] == 0.75
    assert saved["passed"] is True
    assert "Hold instead of hike" in target.with_suffix(".md").read_text()


def test_aws_error_diagnostics_do_not_expose_raw_response():
    from pydantic_ai.exceptions import ModelHTTPError

    from compare import error_details

    exc = ModelHTTPError(
        status_code=403,
        model_name="test",
        body={
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Model is not available for this account. secret-value",
            },
            "ResponseMetadata": {"HTTPHeaders": {"authorization": "secret-token"}},
        },
    )
    details = error_details(exc)
    assert details["http_status"] == 403
    assert details["provider_error_code"] == "AccessDeniedException"
    assert "unavailable to the AWS account" in details["error_hint"]
    assert "secret" not in json.dumps(details)
    row = {
        "case": "test",
        "repeat": 1,
        "name": "Bedrock",
        "outcome": "error",
        "expected": "positive",
        "passed": False,
        "elapsed_seconds": 0.5,
        "requested_model": "bedrock:test",
        "error_stage": "run",
        **details,
    }
    text = markdown_report({"results": [row], "summary": []})
    assert "HTTP 403" in text
    assert "AccessDeniedException" in text
    assert "secret" not in text
