# Transformers Bug Agent

A tool that uses the GitHub Issues API + an open source language model to initiate research on open issues in the `huggingface/transformers` repository by suggesting an initial label and next-step comment.

The pipeline is intentionally simple:

1. **GitHub Issues API** — fetch open issues from a repo.
2. **Issue Summarizer** — convert raw issue JSON into a structured object.
3. **LM advisor** — run a lightweight open-source model to generate an action (`comment`, `pr`, `research-folder`) plus a next-step suggestion
4. **No persistent tracking** — the tool is meant to be rerun per issue and does not store state between runs.

> This project is designed to run on machines with ~8–16GB of RAM. It uses an open-source model (default is a small NLI model) and keeps the GitHub API usage minimal.

## Quick Start

1. Create a Python virtual environment and activate it:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set a GitHub token (required if you hit rate limits):

```bash
export GITHUB_TOKEN=ghp_...
# Windows PowerShell
$env:GITHUB_TOKEN = 'ghp_...'
```

If you see `rate limit exceeded`, set `GITHUB_TOKEN` and re-run.

4. (Optional) Change the model used by the advisor. Default is a lightweight next-token predictor (`distilgpt2`):

```bash
export MODEL_NAME=distilgpt2
# Windows PowerShell
$env:MODEL_NAME = 'distilgpt2'
```

5. Run the agent (defaults to the latest open Transformers issue):

```bash
python -m src.main
```

- To process a specific issue:

```bash
python -m src.main --issue 44593
```

## Notes

- The agent is intentionally simple and focuses on high-level issue assessment (action + next-step suggestion).
- The model can be swapped via the `MODEL_NAME` environment variable.

## Example Labels (from real issues)

- `#44593` → ignore (someone already working on the issue)
- `#44596` → needs-research (reproduce issue)
- `#44485` → comment (references to `vllm` and `sglang`)
