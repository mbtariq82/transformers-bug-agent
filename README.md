# Transformers Bug Agent

A tool using open source LLMs to help maintainers research issues in the `huggingface/transformers` repository by generating either a comment or a PR alongside a notebook explaining any decisions made. All changes are local, the agent is intentionally designed to have a human in the loop.

The pipeline is intentionally simple in order to be efficient:

1. **GitHub Issues API** — fetch open issues from a repo.
2. **Issue Summarizer** — convert raw issue JSON into a structured object.
3. **LM advisor** — run a lightweight open-source model to generate an action (`comment`, `pr`), a detail (comment text or branch name), and an optional research folder path.

> This project is designed to run on machines with ~8–16GB of RAM.

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

4. (Optional) Change the model used by the advisor. Default is `Qwen/Qwen3-1.7b`:

```bash
export MODEL_NAME=Qwen/Qwen3-1.7b
# Windows PowerShell
$env:MODEL_NAME = 'Qwen/Qwen3-1.7b'
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
