"""Command-line interface for Veritas."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .config import Settings
from .engine import Analyst
from .logging_setup import configure_logging
from .mockdata import SAMPLE_QUESTIONS, sample_documents
from .version import __version__


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _seeded_analyst() -> Analyst:
    a = Analyst(Settings())
    for doc in sample_documents():
        a.ingest(doc)
    return a


def _print_answer(ans) -> None:
    print(f"\nQ: {ans.query}")
    tag = "grounded" if ans.grounded else "REFUSED (not in document)"
    print(f"   [{tag}]  agents: {', '.join(ans.agents_used)}")
    print(f"A: {ans.text}")
    if ans.citations:
        cites = "; ".join(c.label() for c in ans.citations)
        print(f"   Sources: {cites}")


def cmd_demo(args) -> int:
    a = _seeded_analyst()
    print(f"Veritas demo — ingested {a.stats()['documents']} documents "
          f"({a.stats()['chunks']} chunks)")
    for q in SAMPLE_QUESTIONS:
        _print_answer(a.ask(q))
    return 0


def cmd_ask(args) -> int:
    a = _seeded_analyst()
    if args.file:
        a.ingest_file(args.file)
    _print_answer(a.ask(args.query))
    return 0


def cmd_serve(args) -> int:
    from .serving.api import run_server

    run_server(host=args.host, port=args.port)
    return 0


def cmd_stats(args) -> int:
    a = _seeded_analyst()
    out = {"stats": a.stats(),
           "sample_answer": a.ask(SAMPLE_QUESTIONS[0]).to_dict()}
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="veritas", description="Domain-specific AI analyst (multi-agent RAG)")
    p.add_argument("--version", action="version", version=f"veritas {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="ingest sample documents and answer sample questions")
    d.set_defaults(func=cmd_demo)

    q = sub.add_parser("ask", help="ask a question (optionally against your own file)")
    q.add_argument("query")
    q.add_argument("--file", default=None, help="also ingest this document (.txt/.md/.pdf)")
    q.set_defaults(func=cmd_ask)

    s = sub.add_parser("stats", help="print engine stats + a sample answer as JSON")
    s.set_defaults(func=cmd_stats)

    sv = sub.add_parser("serve", help="run the FastAPI backend + chat UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _reconfigure_stdout()
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
