"""Entry point for the Transformers Bug Agent."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .issue_advisor import IssueAdvisor
from .github_client import GitHubClient
from .storage import load_seen, mark_seen
from .summarizer import format_issue_text, summarize_issue


LOG = logging.getLogger(__name__)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight issue advisor for a GitHub repo."
    )
    parser.add_argument(
        "--repo",
        default="huggingface/transformers",
        help="Repository to scan (owner/repo). Defaults to huggingface/transformers.",
    )
    parser.add_argument(
        "--issue",
        type=int,
        help="Specific issue number to analyze (defaults to latest open issue).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG if args.verbose else logging.INFO
    )

    # Reduce noisy transformer/huggingface hub logging unless verbose is requested.
    if not args.verbose:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        try:
            # transformers uses its own logging controls; keep it quiet unless requested.
            from transformers import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except Exception:
            pass

    client = GitHubClient()
    advisor = IssueAdvisor()

    seen = set(load_seen())
    LOG.info("Loaded %d seen issues", len(seen))

    issue = None
    try:
        if args.issue is not None:
            LOG.info("Fetching issue #%d", args.issue)
            issue = client.get_issue(args.repo, args.issue)
        else:
            LOG.info("Fetching latest open issue")
            issue = client.get_latest_issue(args.repo)

        if issue is None:
            LOG.warning(
                "No issue found to analyze (this can happen if the repo returns only pull requests or has no matching issues)."
            )
            return 0
    except Exception as exc:  # pragma: no cover
        # GitHub rate limits are common when unauthenticated.
        msg = str(exc)
        if getattr(exc, "response", None) is not None and exc.response.status_code == 403:
            remaining = exc.response.headers.get("X-RateLimit-Remaining")
            reset = exc.response.headers.get("X-RateLimit-Reset")
            LOG.error("GitHub rate limit exceeded (remaining=%s).", remaining)
            LOG.error(
                "Set GITHUB_TOKEN in your environment to avoid rate limits, or wait until rate limits reset."
            )
            if reset:
                LOG.error("Rate limit resets at UNIX epoch seconds: %s", reset)
        else:
            LOG.error("Failed to fetch issue: %s", msg)
        return 1

    structured = summarize_issue(issue)
    number = structured.get("number")

    # If the user explicitly requested an issue, always process it (even if seen).
    if args.issue is None and number in seen:
        LOG.info("Issue #%d already seen, exiting", number)
        return 0

    prompt = format_issue_text(structured)
    if not prompt:
        LOG.warning("Issue #%s has no text to analyze", number)
        return 0

    result = advisor.advise(prompt)

    LOG.info(
        "#%s %s => %s (%.2f)",
        number,
        structured.get("title"),
        result.get("action"),
        result.get("score"),
    )

    print("---")
    print(f"#{number} {structured.get('title')}")
    print(f"URL: {structured.get('url')}")
    print(f"Action: {result.get('action')} ({result.get('score'):.2f})")
    if result.get("next_steps"):
        print(f"Next steps: {result.get('next_steps')}")
    print()

    new_seen = [number]

    if new_seen:
        mark_seen(new_seen)
        LOG.info("Marked %d new issues as seen", len(new_seen))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
