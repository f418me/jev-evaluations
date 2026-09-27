# Jev Evaluations

An executable evaluation of **Jev through TypeSafe and Pydantic AI** using FOMC announcements: define structured answers, provide expectations and statement text, inspect field confidence, and derive a traceable decision in Python.

The project compares model responses to an FOMC statement against a supplied expectation baseline. It includes five synthetic scenarios and reports decisions, confidence, explanations, and response times. It does not collect market prices, calculate trading returns, or execute trades.

## Quick start

Requirements: Python ≥ 3.12, `uv`, and a TypeSafe API key for live requests.

```bash
uv sync --locked
cp -n .env.example .env
```

Set `TYPESAFE_API_KEY` in `.env`, preserving any existing configuration:

```bash
# Evaluate one synthetic case with Jev.
uv run python fomc.py

# All five cases, using only Jev, with a report.
uv run python compare.py --jev-only

# Compare Jev with the LLMs in models.json; requires their provider keys.
uv run python compare.py --case all
```

Live requests send expectations and statement text to the configured provider. `JEV_MODEL` controls `fomc.py`; `compare.py` reads model IDs from `models.json`. The default is `typesafe:jev-latest`. Use a returned versioned model ID for reproducible reruns.

## How the example works

1. `Baseline` and `Statement` validate expectations and announcement text.
2. `build_state()` combines the expectation baseline and statement text, excluding expected-result labels.
3. `FOMCJudgments` defines questions about the rate change, tone, and guidance relative to expectations.
4. `build_agent()` calls Jev through the TypeSafe adapter.
5. `decide()` applies a Python policy to answers and required field confidence.

A smaller hike or larger cut than expected produces `positive`; the opposite surprise produces `negative`. When the rate change matches expectations, guidance determines the result. Missing or insufficient confidence in required answers produces `abstain`. The default threshold of 0.8 is an experimental setting, not a validated trading rule.

| Synthetic case | Expected result |
|---|---|
| `expected-hike` | `neutral` |
| `surprise-hold` | `positive` |
| `hawkish-50bp` | `negative` |
| `dovish-hike` | `positive` |
| `preview` | `abstain` |

```bash
uv run python compare.py --jev-only --case surprise-hold hawkish-50bp --repeat 3
uv run python fomc.py --case all --output results/fomc.json
```

Reports include model decisions, expected scenario labels, field confidence, model versions, and elapsed time. PASS/FAIL compares a model decision with the manually defined scenario target; it does not measure subsequent market performance. Exit code 1 indicates an error or a target mismatch. Local reports are stored in `results/` and excluded from Git.

## Baseline and statement inputs

The included baseline is `data/baseline.json`; the five statement fixtures are in `data/statements/`. Expected scenario labels are stored separately in `data/cases.json`. These fixtures are synthetic examples, not a live market feed.

To evaluate your own current statement with Jev, supply a JSON payload with a `content` field and a matching expectation baseline:

```bash
uv run python fomc.py \
  --baseline /absolute/path/baseline.json \
  --statement /absolute/path/statement.json \
  --output results/current-statement.json
```

Custom statements have no predefined scenario target, so no PASS/FAIL score is assigned. Multi-model comparisons through `compare.py` use the configured scenarios in `data/cases.json`.

## Project layout and checks

| File or directory | Purpose |
|---|---|
| `fomc.py` | Jev schemas, model inputs, and decision policy |
| `compare.py` | Repeated tests and optional comparisons with other models |
| `llm.py` | Original LLM approach for comparison |
| `data/` | Expectation baseline, statement fixtures, scenario targets, and original LLM prompt |
| `models.json`, `models.bedrock.json` | Optional model configurations without credentials |
| `tests/` | Offline decision-policy, provider-adapter, and model-comparison tests |
| `docs/evaluation.md` | Confidence, provenance, custom inputs, and provider details |

```bash
uv run pytest -q
uv run ruff check .
```

Tests use simulated responses without API keys. They check implementation rather than live model quality. Keep `uv.lock`, `.env.example`, source, tests, and data in version control. `.env`, other local environment files, `results/`, virtual environments, IDE files, and caches are ignored.

See [evaluation details](docs/evaluation.md) for Asentrx provenance and optional OpenAI, Groq, Gemini, and Bedrock comparisons. Those comparisons use different decision policies and are not pure model benchmarks.
