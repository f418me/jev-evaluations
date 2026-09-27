import json

import httpx2
import pytest
from pydantic import ValidationError
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from fomc import (
    ROOT,
    Baseline,
    FOMCJudgments,
    Statement,
    analyze,
    build_agent,
    build_state,
    decide,
)

CASES = json.loads((ROOT / "data/cases.json").read_text())
# Scripted API answers, NOT evidence of Jev classification quality.
ANSWERS = {
    "expected-hike": (25, "hawkish", "in_line"),
    "surprise-hold": (0, "dovish", "more_dovish"),
    "hawkish-50bp": (50, "hawkish", "more_hawkish"),
    "dovish-hike": (25, "dovish", "more_dovish"),
    "preview": (None, "neutral", "unknown"),
}


@pytest.fixture
def baseline():
    return Baseline.model_validate_json((ROOT / "data/baseline.json").read_text())


def judgments(change=0, guidance="more_dovish"):
    return FOMCJudgments(
        actual_change_bp=change,
        actual_narrative="dovish",
        guidance_vs_baseline=guidance,
    )


def confidence(value=0.95):
    return {"confidence": dict.fromkeys(FOMCJudgments.model_fields, value)}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_real_typesafe_adapter_with_mock_http(case, baseline):
    change, tone, guidance = ANSWERS[case["name"]]
    requests = []

    def respond(request):
        assert request.url.host == "api.typesafe.ai"
        body = json.loads(request.content)
        assert body["model"] == "jev-latest"
        requests.append(body)
        questions = body["questions"]
        assert set(questions) == set(FOMCJudgments.model_fields)
        assert "is_new_rate_decision" not in questions
        answers = {}
        for name, value in {
            "actual_change_bp": change,
            "actual_narrative": tone,
            "guidance_vs_baseline": guidance,
        }.items():
            options = questions[name]["criteria"]
            label = (
                str(value)
                if value is not None
                else next(
                    k
                    for k in options
                    if k
                    not in {str(i) for i in [-100, -75, -50, -25, 0, 25, 50, 75, 100]}
                )
            )
            assert label in options
            answers[name] = {
                "type": "choice",
                "choice": label,
                "confidence": 0.95,
                "probabilities": {
                    k: 0.99 if k == label else 0.01 / (len(options) - 1)
                    for k in options
                },
            }
        return httpx2.Response(
            200,
            json={
                "model": "jev-mock-test",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    model = TypeSafeModel(
        "jev-latest", provider=TypeSafeProvider(api_key="test-only", http_client=client)
    )
    statement = Statement.model_validate_json(
        (ROOT / "data/statements" / case["file"]).read_text()
    )
    report = analyze(build_agent(model), baseline, statement, 0.8)
    decision = report["decision"]
    observed = (
        decision["impact_on_bitcoin"]
        if decision["status"] == "assessed"
        else decision["status"]
    )
    assert observed == case["expected"]
    assert report["model"] == "jev-mock-test"
    assert len(requests) == 1
    # Actual outbound request must not leak gold labels through metadata or source notes.
    state = json.dumps(requests[0]["state"])
    assert case["file"] not in state
    assert "content-id" not in state
    assert baseline.notes not in state


@pytest.mark.parametrize(
    "change,guidance,expected",
    [
        (25, "in_line", "neutral"),
        (0, "in_line", "positive"),
        (50, "in_line", "negative"),
        (25, "more_dovish", "positive"),
        (25, "more_hawkish", "negative"),
        (50, "more_dovish", "negative"),
        (0, "more_hawkish", "positive"),
        (0, "mixed", "positive"),
    ],
)
def test_surprise_policy(baseline, change, guidance, expected):
    assert (
        decide(judgments(change, guidance), baseline, confidence()).impact_on_bitcoin
        == expected
    )


@pytest.mark.parametrize(
    "value", [0.79, None, float("nan"), float("inf"), -1, 1.1, True, "0.95"]
)
def test_low_missing_or_invalid_confidence_abstains(baseline, value):
    details = confidence()
    details["confidence"]["actual_change_bp"] = value
    assert decide(judgments(), baseline, details).status == "abstain"


def test_absent_provider_metadata_abstains(baseline):
    assert decide(judgments(), baseline, None).status == "abstain"


@pytest.mark.parametrize("change,guidance", [(None, "in_line"), (25, "unknown")])
def test_unknowns_are_not_neutral(baseline, change, guidance):
    assert (
        decide(judgments(change, guidance), baseline, confidence()).status == "abstain"
    )


@pytest.mark.parametrize("change,expected", [(0, "positive"), (50, "negative")])
@pytest.mark.parametrize(
    "guidance", ["more_dovish", "more_hawkish", "mixed", "unknown", "in_line"]
)
@pytest.mark.parametrize("guidance_confidence", [None, 0.1, float("nan")])
def test_rate_surprise_ignores_guidance_and_its_confidence(
    baseline, change, expected, guidance, guidance_confidence
):
    details = {"confidence": {"actual_change_bp": 1.0}}
    if guidance_confidence is not None:
        details["confidence"]["guidance_vs_baseline"] = guidance_confidence
    result = decide(judgments(change, guidance), baseline, details)
    assert result.status == "assessed"
    assert result.impact_on_bitcoin == expected
    assert result.minimum_field_confidence == 1.0
    assert result.rate_surprise_bp == change - 25


@pytest.mark.parametrize("value", [None, 0.1, float("nan")])
def test_matching_rate_still_requires_guidance_confidence(baseline, value):
    details = confidence()
    details["confidence"]["guidance_vs_baseline"] = value
    result = decide(judgments(25), baseline, details)
    assert result.status == "abstain"
    assert result.rate_surprise_bp == 0


def test_boundary_and_baseline_change(baseline):
    assert decide(judgments(), baseline, confidence(0.8)).status == "assessed"
    hold_baseline = Baseline(
        **{
            **baseline.model_dump(),
            "expected_interest_rate_change_type": "hold",
            "expected_interest_rate_change_amount": "0bp",
        }
    )
    assert (
        decide(judgments(0, "in_line"), hold_baseline, confidence()).impact_on_bitcoin
        == "neutral"
    )
    assert (
        decide(judgments(0, "in_line"), baseline, confidence()).impact_on_bitcoin
        == "positive"
    )


def test_contradictory_baseline_rejected(baseline):
    with pytest.raises(ValidationError):
        Baseline(
            **{**baseline.model_dump(), "expected_interest_rate_change_amount": "-25bp"}
        )


def test_state_separates_baseline_and_announcement(baseline):
    state = json.loads(build_state(baseline, Statement(content="No decision yet.")))
    assert state["announcement"] == "No decision yet."
    assert (
        state["pre_announcement_expectations"]["expected_interest_rate_change_amount"]
        == "+25bp"
    )
    assert "notes" not in state["pre_announcement_expectations"]
