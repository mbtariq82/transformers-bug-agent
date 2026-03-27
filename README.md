# Bug Agent
An autonomous agent to help maintainers research issues/bugs. All changes are local, the agent is intentionally designed to have a human in the loop.

Pipeline:
1. **GitHub Issues API** — fetch open issues from a repo.
2. **Issue Summarizer** — convert raw issue JSON into a structured object.
3. **LM advisor** — use SmolAgents with a lightweight open-source model (default: HuggingFaceTB/SmolLM-1.7B) to analyze the issue and provide guidance. Current code always prefers CodeAgent with tools (fallback to direct `model.generate()` only on CodeAgent failure).


## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start
The default model `HuggingFaceTB/SmolLM-1.7B` requires ~4GB RAM. For best results, run on Google Colab (free tier has 16GB) or a machine with sufficient memory.

(Optional) Change the model used by the advisor. Default is `HuggingFaceTB/SmolLM-1.7B`:

```bash
export MODEL_NAME=HuggingFaceTB/SmolLM-1.7B
# Windows PowerShell
$env:MODEL_NAME = 'HuggingFaceTB/SmolLM-1.7B'
```

Run the agent (defaults to the latest open Transformers issue):

```bash
python -m src.main
```

- To process a specific issue:

```bash
python -m src.main --issue 44593
```

## Example Labels (from Transformers)

- `#44593`, `#44910` → check PR
- `#44485` → references to `vllm` and `sglang`
- `#44829` → need GPU
- `#44869` → existing issue/PR
- `#44912` → start by recreating issue
- `#44995` → PR #44950 will fix this
