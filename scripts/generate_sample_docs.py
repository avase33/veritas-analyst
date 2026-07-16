#!/usr/bin/env python3
"""Write the sample documents to disk and (optionally) run a few queries.

    python scripts/generate_sample_docs.py            # write .txt files + demo Q&A
    python scripts/generate_sample_docs.py --out docs/ # write files only
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Generate sample documents for Veritas")
    ap.add_argument("--out", default="sample_docs", help="directory to write .txt files")
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args(argv)

    from veritas.mockdata import sample_documents, SAMPLE_QUESTIONS

    os.makedirs(args.out, exist_ok=True)
    docs = sample_documents()
    for doc in docs:
        path = os.path.join(args.out, (doc.source or doc.title.replace(" ", "_") + ".txt"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.text)
        print(f"wrote {path}")

    if args.no_demo:
        return 0

    from veritas.config import Settings
    from veritas.engine import Analyst

    a = Analyst(Settings())
    for doc in docs:
        a.ingest(doc)
    print(f"\nIngested {a.stats()['documents']} docs ({a.stats()['chunks']} chunks). Sample Q&A:\n")
    for q in SAMPLE_QUESTIONS[:3]:
        ans = a.ask(q)
        print(f"Q: {q}")
        print(f"A: {ans.text}")
        if ans.citations:
            print("   Sources: " + "; ".join(c.label() for c in ans.citations))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
