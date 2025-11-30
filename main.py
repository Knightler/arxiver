# src/main.py
# type: ignore

from __future__ import annotations
import os
import re
from datetime import datetime

from src.state import graph
from src.graph import answer

"""
Main entry point for the Arxiver application.

The goal for this project is to build a system that can fetch the latest papers from the Arxiv platform,
explain their content in plain-language terms,
and answer user questions about them.

The intention is to be able to read multiple papers as fast as possible, and to always be up-to-date with the latest research.

version => 0.2.1

Future updates may include:
- Support for more research platforms (e.g., PubMed, IEEE Xplore) => version 0.3.0
- Enhanced explanation capabilities (e.g., visual summaries) => version 0.4.0
- Improved question-answering accuracy and context-awareness => version 0.5.0
- User interface enhancements (e.g., web app, mobile app) => version 0.6.0
... etc.
"""

"""
Interactive CLI for Arxiver v0.2

Flow:
1. Ask user for a topic / query.
2. Fetch the latest relevant paper from arXiv.
3. Show:
   - Title
   - Authors
   - PDF URL (if available)
   - Plain-language explanation
4. Enter a Q&A loop:
   - User asks questions about the paper.
   - Model answers based on the paper.
   - Exit only when user types 'quit' / 'q' / 'exit'.
"""

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "paper"


def save_initial_markdown(path: str, paper: dict, notes_md: str) -> None:
    header = (
        f"# {paper['title']}\n\n"
        f"- **Topic/Query:** {paper.get('topic','')}\n"
        f"- **Authors:** {', '.join(paper.get('authors', []))}\n"
        f"- **arXiv ID:** {paper.get('arxiv_id','')}\n"
        f"- **PDF:** {paper.get('pdf_url','')}\n"
        f"- **Saved at:** {datetime.now().isoformat(timespec='seconds')}\n\n"
        "---\n\n"
        "## Explanation / Study Notes\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(notes_md.strip())
        f.write("\n\n---\n\n## Q&A Log\n\n")


def append_qa(path: str, q: str, a: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"### Q: {q}\n\n")
        f.write(a.strip())
        f.write("\n\n")


def main() -> None:
    topic = input("Enter a research topic or a paper title: ").strip()

    try:
        result = graph.invoke({"topic": topic})
    except Exception as e:
        print(f"Error fetching paper: {e}")
        return

    paper = result["paper"]
    notes_md = result["notes_md"]

    # ---- print ----
    print("\n=== Topic / Query ===")
    print(topic)

    print("\n=== Paper Details ===")
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper.get('authors', []))}")
    print(f"PDF URL: {paper.get('pdf_url', '')}")

    print("\n=== Explanation ===")
    print(notes_md)

    # ---- save markdown ----
    os.makedirs("papers", exist_ok=True)
    safe = slugify(paper["title"])
    aid = paper.get("arxiv_id", "") or "noid"
    out_path = os.path.join("papers", f"{safe}__{aid}.md")
    save_initial_markdown(out_path, paper, notes_md)
    print(f"\n[Saved notes to] {out_path}")

    # ---- interactive Q&A ----
    print("\n=== Q&A Session ===")
    print("Ask questions about the paper. Type 'quit', 'q', or 'exit' to end.\n")

    while True:
        q = input("Your question: ").strip()
        if q.lower() in {"quit", "q", "exit"}:
            print("bye")
            break
        if not q:
            continue

        a = answer(paper, notes_md, q)
        print("\n" + a + "\n")
        append_qa(out_path, q, a)


if __name__ == "__main__":
    main()
