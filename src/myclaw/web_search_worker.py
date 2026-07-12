"""Isolated process entry point for the synchronous DuckDuckGo SDK."""

import json
import sys

from myclaw.web_search import _duckduckgo_text_search


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 2:
        return 2
    query, raw_max_results = values
    try:
        max_results = int(raw_max_results)
    except ValueError:
        return 2
    try:
        results = _duckduckgo_text_search(query, max_results=max_results)
    except Exception:
        return 1
    payload = json.dumps(results, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
