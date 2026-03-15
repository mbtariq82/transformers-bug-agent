# Transformers Bug Agent

A tool using open source LLMs to help maintainers research issues in the `huggingface/transformers` repository by generating either a comment or a PR alongside a notebook. All changes are local, the agent is intentionally designed to have a human in the loop.

Pipeline:
1. **GitHub Issues API** — fetch open issues from a repo.
2. **Issue Summarizer** — convert raw issue JSON into a structured object.
3. **LM advisor** — run a lightweight open-source model to generate an action (`comment`, `pr`), a detail (comment text or branch name), and a research notebook path.

> This project is designed to run on machines with ~8–16GB of RAM.

## Quick Start
(Optional) Change the model used by the advisor. Default is `Qwen/Qwen3-1.7b`:

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

## Example Labels (from real issues)

- `#44593` → ignore (someone already working on the issue)
- `#44596` → needs-research (reproduce issue)
- `#44485` → comment (references to `vllm` and `sglang`)
