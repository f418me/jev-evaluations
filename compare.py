"""Compare Jev policy with the original Asentrx LLM decision logic."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from fomc import ROOT, Baseline, Statement, build_agent, build_state, decide
from llm import FEDDecisionImpact, IrrelevantFEDContent, build_llm_agent


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    kind: Literal["jev", "llm"]
    model: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_provider(self):
        valid = (self.kind == "jev" and self.model.startswith("typesafe:")) or (
            self.kind == "llm" and not self.model.startswith("typesafe:")
        )
        if not valid:
            raise ValueError("Use typesafe: for jev and an LLM provider for llm")
        return self


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ModelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self):
        if len({s.name for s in self.models}) != len(self.models):
            raise ValueError("Model names must be unique")
        return self


def create_agent(spec, baseline):
    return (
        build_agent(spec.model)
        if spec.kind == "jev"
        else build_llm_agent(spec.model, baseline)
    )


def evaluate(agent, spec, baseline, statement, threshold):
    prompt = (
        build_state(baseline, statement) if spec.kind == "jev" else statement.content
    )
    start = perf_counter()
    result = agent.run_sync(prompt)
    elapsed = perf_counter() - start
    output = result.output
    if spec.kind == "jev":
        decision = decide(output, baseline, result.response.provider_details, threshold)
        status = decision.status
        outcome = decision.impact_on_bitcoin or status
        confidence = decision.minimum_field_confidence
        reasoning = decision.reasoning
    elif isinstance(output, FEDDecisionImpact):
        status, outcome = "assessed", output.impact_on_bitcoin
        confidence, reasoning = output.confidence, output.reasoning
    elif isinstance(output, IrrelevantFEDContent):
        status = outcome = "irrelevant"
        confidence, reasoning = None, output.reason
    else:
        status = outcome = "failed"
        confidence, reasoning = None, output.error_message
    return {
        "resolved_model": result.response.model_name,
        "elapsed_seconds": round(elapsed, 6),
        "status": status,
        "outcome": outcome,
        "confidence": confidence,
        "confidence_kind": "required_field_minimum"
        if spec.kind == "jev"
        else "llm_signal_self_assessment",
        "reasoning": reasoning,
        "raw_output": output.model_dump(),
        "provider_details": result.response.provider_details,
        "request_count": result.usage.requests,
    }


def error_details(exc):
    """Expose diagnostic codes and fixed hints, never raw bodies, headers or prompts."""
    result = {"error_type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        body = getattr(exc, "response", {})
    error = body.get("Error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        return result
    code = error.get("Code")
    allowed_codes = {
        "AccessDeniedException",
        "ValidationException",
        "ResourceNotFoundException",
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "UnrecognizedClientException",
        "ExpiredTokenException",
    }
    if code in allowed_codes:
        result["provider_error_code"] = code
    message = str(error.get("Message", "")).lower()
    if "not available for this account" in message:
        hint = "This model is unavailable to the AWS account. AWS refers additional access options to sales."
    elif "retention" in message or "aws_review" in message:
        hint = "The model requires an appropriate AWS data-retention approval."
    elif "model identifier is invalid" in message:
        hint = "AWS does not accept this model ID in the selected region. Check the model ID and inference profile."
    elif "use case" in message or "use-case" in message:
        hint = (
            "AWS requires approval of Anthropic use-case information for this account."
        )
    elif (
        "marketplace" in message or "subscription" in message or "agreement" in message
    ):
        hint = "Check AWS model access, Marketplace activation, and permissions."
    elif code == "AccessDeniedException":
        hint = "Check AWS model access, IAM permissions, and region."
    elif code == "ThrottlingException":
        hint = "AWS request limit reached. Retry later or check the quota."
    else:
        hint = None
    if hint:
        result["error_hint"] = hint
    return result


def run_comparison(
    config, baseline, cases, repeats=1, threshold=0.8, agent_factory=create_agent
):
    """Sequential calls; rotate model order between repeats, fresh history per call."""
    agents, setup_errors = {}, {}
    for spec in config.models:
        try:
            agents[spec.name] = agent_factory(spec, baseline)
        except Exception as exc:  # noqa: BLE001 -- missing keys must not cancel other models
            setup_errors[spec.name] = error_details(exc)
    rows = []
    for repeat in range(1, repeats + 1):
        offset = (repeat - 1) % len(config.models)
        ordered = config.models[offset:] + config.models[:offset]
        for case in cases:
            statement = Statement.model_validate_json(
                (ROOT / "data/statements" / case["file"]).read_text()
            )
            for spec in ordered:
                expected = case["expected"]
                # Preview is abstain in the Jev policy and irrelevant in the original agent.
                if spec.kind == "llm":
                    expected = case.get("llm_expected", expected)
                row = {
                    "case": case["name"],
                    "repeat": repeat,
                    "name": spec.name,
                    "kind": spec.kind,
                    "requested_model": spec.model,
                    "expected": expected,
                }
                if spec.name in setup_errors:
                    row.update(
                        status="error",
                        outcome="error",
                        elapsed_seconds=None,
                        **setup_errors[spec.name],
                        error_stage="setup",
                    )
                else:
                    start = perf_counter()
                    try:
                        row.update(
                            evaluate(
                                agents[spec.name], spec, baseline, statement, threshold
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 -- preserve partial comparison results
                        row.update(
                            status="error",
                            outcome="error",
                            error_stage="run",
                            **error_details(exc),
                            elapsed_seconds=round(perf_counter() - start, 6),
                        )
                row["passed"] = row["outcome"] == expected
                rows.append(row)
    summaries = []
    for spec in config.models:
        selected = [r for r in rows if r["name"] == spec.name]
        times = [
            r["elapsed_seconds"]
            for r in selected
            if r["status"] not in ("error", "failed")
        ]
        summaries.append(
            {
                "name": spec.name,
                "runs": len(selected),
                "passed": sum(r["passed"] for r in selected),
                "errors": sum(r["status"] in ("error", "failed") for r in selected),
                "mean_seconds": mean(times) if times else None,
                "median_seconds": median(times) if times else None,
            }
        )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "synthetic_fixtures": True,
        "config": config.model_dump(),
        "baseline": baseline.model_dump(),
        "jev_threshold": threshold,  # retained for existing report consumers
        "provider_confidence_threshold": threshold,
        "repeats": repeats,
        "results": rows,
        "summary": summaries,
    }


def markdown_report(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def seconds(value):
        return "—" if value is None else f"{value:.3f}"

    jev_only = bool(report.get("config", {}).get("models")) and all(
        model["kind"] == "jev" for model in report["config"]["models"]
    )
    lines = [
        "# FOMC: Jev case tests" if jev_only else "# FOMC: Jev vs. original LLM policy",
        "",
        "Synthetic cases. Jev prioritizes rate surprises."
        if jev_only
        else "Synthetic cases. Jev prioritizes rate surprises; LLMs use the original Asentrx prompt.",
        "Timing includes the agent call, network, validation, and retries; excludes agent initialization.",
        "Confidence concerns required field answers, not the probability of profit."
        if jev_only
        else "Confidence scores have different meanings and are not directly comparable.",
        "",
        "| Case | Repeat | Model | Decision | Expected | Test | Seconds | Confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        values = [
            r["case"],
            r["repeat"],
            r["name"],
            r["outcome"],
            r["expected"],
            "PASS" if r["passed"] else "FAIL",
            seconds(r["elapsed_seconds"]),
            r.get("confidence", "—"),
        ]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += [
        "",
        "## Summary",
        "",
        "| Model | Passed | Errors | Mean (s) | Median (s) |",
        "|---|---|---|---|---|",
    ]
    for s in report["summary"]:
        lines.append(
            f"| {cell(s['name'])} | {s['passed']}/{s['runs']} | {s['errors']} | {seconds(s['mean_seconds'])} | {seconds(s['median_seconds'])} |"
        )
    lines += [
        "",
        "Summary timings exclude errors and FailedFEDAnalysis results.",
        "",
        "## Individual decisions",
        "",
    ]
    for r in report["results"]:
        lines += [
            f"### {cell(r['case'])} · {cell(r['name'])} · Repeat {r['repeat']}",
            "",
            f"Requested: `{cell(r['requested_model'])}` · Resolved model: `{cell(r.get('resolved_model', '—'))}`",
            "",
            cell(
                r.get(
                    "reasoning",
                    f"Error ({r.get('error_stage')}): {r.get('error_type')} "
                    f"· HTTP {r.get('http_status', '—')} "
                    f"· {r.get('provider_error_code', '—')} "
                    f"{r.get('error_hint', '')}",
                )
            ),
            "",
        ]
    return "\n".join(lines)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "models.json")
    parser.add_argument(
        "--jev-only", action="store_true", help="Run only Jev entries from the config"
    )
    parser.add_argument(
        "--case", nargs="+", action="extend", help="Case names (default: all)"
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/baseline.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Jev provider confidence gate"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or not 0 <= args.threshold <= 1:
        parser.error("--repeat must be positive and --threshold between 0 and 1")
    config = ComparisonConfig.model_validate_json(args.config.read_text())
    if args.jev_only:
        kind = "jev"
        models = [model for model in config.models if model.kind == kind]
        if not models:
            parser.error(f"No {kind} model in the selected config")
        config = ComparisonConfig(models=models)
    if args.output is None:
        args.output = (
            ROOT / "results" / ("jev.json" if args.jev_only else "comparison.json")
        )
    baseline = Baseline.model_validate_json(args.baseline.read_text())
    cases = json.loads((ROOT / "data/cases.json").read_text())
    requested = set(args.case or ["all"])
    unknown = requested - {"all"} - {case["name"] for case in cases}
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(sorted(unknown))}")
    cases = [c for c in cases if "all" in requested or c["name"] in requested]
    report = run_comparison(config, baseline, cases, args.repeat, args.threshold)
    rendered = markdown_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable_python)
        + "\n"
    )
    markdown_path = args.output.with_suffix(".md")
    if markdown_path == args.output:
        markdown_path = args.output.with_name(args.output.name + ".md")
    markdown_path.write_text(rendered + "\n")
    print(rendered)
    return int(any(not r["passed"] for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
