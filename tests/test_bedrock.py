import pytest
from botocore.stub import Stubber

from compare import ComparisonConfig, ModelSpec, create_agent, evaluate
from fomc import ROOT, Baseline, Statement


@pytest.fixture
def aws_environment(monkeypatch):
    # Never use real credentials, profiles, bearer tokens or metadata services.
    for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-only")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-only")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-only")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.mark.parametrize(
    "tool_name,args,outcome",
    [
        (
            "FEDDecisionImpact",
            {
                "impact_on_bitcoin": "positive",
                "confidence": 0.9,
                "reasoning": "Dovish hike",
                "actual_fed_decision_summary": "Hike with dovish guidance",
            },
            "positive",
        ),
        ("IrrelevantFEDContent", {"reason": "Preview"}, "irrelevant"),
        ("FailedFEDAnalysis", {"error_message": "Unclear text"}, "failed"),
    ],
)
def test_bedrock_converse_end_to_end(aws_environment, tool_name, args, outcome):
    baseline = Baseline.model_validate_json((ROOT / "data/baseline.json").read_text())
    spec = ModelSpec(
        name="Nova", kind="llm", model="bedrock:amazon.nova-lite-v1:0"
    )
    agent = create_agent(spec, baseline)
    captured = []

    def inspect_request(params, **kwargs):
        captured.append(params)

    agent.model.client.meta.events.register(
        "before-parameter-build.bedrock-runtime.Converse", inspect_request
    )
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "test-call",
                            "name": "final_result_" + tool_name,
                            "input": args,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 100, "outputTokens": 25, "totalTokens": 125},
        "metrics": {"latencyMs": 12},
    }
    with Stubber(agent.model.client) as stub:
        stub.add_response("converse", response)
        result = evaluate(
            agent, spec, baseline, Statement(content="Statement only"), 0.8
        )
        stub.assert_no_pending_responses()
    assert result["outcome"] == outcome
    assert result["elapsed_seconds"] >= 0
    assert result["request_count"] == 1
    assert captured[0]["modelId"] == "amazon.nova-lite-v1:0"
    assert captured[0]["messages"][0]["content"] == [{"text": "Statement only"}]
    assert baseline.notes in captured[0]["system"][0]["text"]


def test_bedrock_example_constructs_all_agents(aws_environment, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    config = ComparisonConfig.model_validate_json(
        (ROOT / "models.bedrock.json").read_text()
    )
    baseline = Baseline.model_validate_json((ROOT / "data/baseline.json").read_text())
    for spec in config.models:
        assert create_agent(spec, baseline) is not None
