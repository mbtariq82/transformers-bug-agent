"""Entry point for the Transformers Bug Agent."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .issue_advisor import IssueAdvisor
from .github_client import GitHubClient
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
    #logging.basicConfig(
    #    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG if args.verbose else logging.INFO
    #)

    client = GitHubClient()
    advisor = IssueAdvisor()

    issue = None
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
    
    structured = summarize_issue(issue)
    number = structured.get("number")


    prompt = format_issue_text(structured)
    if not prompt:
        LOG.warning("Issue #%s has no text to analyze", number)
        return 0

    response = advisor.advise(prompt, number)

    # Log a short preview of the response for quick debugging.
    LOG.info(
        "#%s %s => %s",
        number,
        structured.get("title"),
        response.replace("\n", " ")[:80],
    )

    print("---")
    print(f"#{number} {structured.get('title')}")
    print(f"URL: {structured.get('url')}")
    print("Response:")
    print(response)
    print()

    # No persistent state is tracked; rerun as needed.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
