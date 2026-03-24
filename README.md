# Bug Agent
An autonomous agent to help maintainers research issues/bugs. All changes are local, the agent is intentionally designed to have a human in the loop.

Pipeline:
1. **GitHub Issues API** — fetch open issues from a repo.
2. **Issue Summarizer** — convert raw issue JSON into a structured object.
3. **LM advisor** — use SmolAgents with a lightweight open-source model (default: HuggingFaceTB/SmolLM-1.7B) to analyze the issue and provide guidance. For small-context/token-length models, the code runs a direct `model.generate()` path; for larger context models, it prefers CodeAgent tooling. (Current code does branch on `tokenizer.model_max_length` at runtime.)


## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start
(Optional) Change the model used by the advisor. Default is `gpt2`:

```bash
export MODEL_NAME=Qwen/Qwen3-1.7b
# Windows PowerShell
$env:MODEL_NAME = 'Qwen/Qwen3-1.7b'
```

Run the agent (defaults to the latest open Transformers issue):

```bash
python -m src.main
```

- To process a specific issue:

```bash
python -m src.main --issue 44593
```

## Example Labels (from transformers)

- `#44593`, `#44910` → check PR
- `#44485` → references to `vllm` and `sglang`
- `#44829` → need GPU
- `#44869` → existing issue/PR
- `#44912` → start by recreating issue