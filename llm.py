"""Frozen Asentrx FED output schemas and prompt; no trading or application imports."""

from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from fomc import ROOT, Baseline


class FEDDecisionImpact(BaseModel):
    """Output model for the FED decision analysis agent."""

    impact_on_bitcoin: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str
    actual_fed_decision_summary: str
    actual_interest_rate_change_type: Literal["increase", "decrease", "hold"] | None = (
        None
    )
    actual_interest_rate_change_amount: str | None = None
    actual_narrative: Literal["hawkish", "dovish", "neutral", "mixed"] | None = None


class FailedFEDAnalysis(BaseModel):
    """Indicates that the FED decision analysis failed."""

    error_message: str


class IrrelevantFEDContent(BaseModel):
    """
    Indicates that the provided content is not relevant to a FED interest rate decision.
    """

    reason: str


def build_llm_agent(model: str | Model, baseline: Baseline) -> Agent:
    prompt = (ROOT / "data/llm_fed_prompt.txt").read_text().strip()
    return Agent(
        model,
        output_type=FEDDecisionImpact | IrrelevantFEDContent | FailedFEDAnalysis,
        system_prompt=prompt.replace(
            "{baseline_json}", baseline.model_dump_json(indent=2)
        ),
        model_settings={"timeout": 30},
    )
