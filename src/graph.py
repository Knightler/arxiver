# src/graph.py
# type: ignore
"""
Graph structure for the Arxiver application.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import arxiv
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


def fetch(topic: str) -> Dict[str, Any]:
    """
    Fetches the latest paper from arXiv related to the query.
    """
    def run_search(q: str, max_results: int = 1):
        search = arxiv.Search(
            query=f'all:"{q}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        return list(search.results())

    results = run_search(topic)

    # ✅ FIX: if not results, not if results
    if not results:
        raise ValueError("No papers found for the given query.")

    paper = results[0]

    # best-effort arxiv id
    entry_id = getattr(paper, "entry_id", "") or ""
    arxiv_id = ""
    if entry_id:
        # e.g. http://arxiv.org/abs/1706.03762v7 -> 1706.03762v7
        arxiv_id = entry_id.rstrip("/").split("/")[-1]

    return {
        "topic": topic,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "pdf_url": getattr(paper, "pdf_url", "") or "",
        "entry_id": entry_id,
        "arxiv_id": arxiv_id,
        "published": str(getattr(paper, "published", "") or ""),
        "updated": str(getattr(paper, "updated", "") or ""),
    }


def explain(paper: Dict[str, Any]) -> str:
    """
    Generates ultra beginner-friendly, high-detail notes in Markdown.
    IMPORTANT: based on Title+Abstract only (no hallucinations).
    """

    system = SystemMessage(
        content=(
            "You are an expert research tutor.\n"
            "Goal: produce ultra-clear, beginner-friendly study notes in Markdown.\n"
            "Constraints:\n"
            "- Do NOT invent details not supported by the provided Title/Abstract.\n"
            "- If a detail is unknown (methods, hyperparams, datasets, exact results), explicitly mark it as 'Not specified in abstract'.\n"
            "- Explain concepts from the ground up.\n"
            "- Prefer concrete intuition + step-by-step reasoning + simple examples.\n"
        )
    )

    human = HumanMessage(
        content=(
            "Create **complete study notes** for this paper.\n\n"
            "What I want:\n"
            "- NOT a short summary. I want thorough notes that make reading the full paper easy.\n"
            "- Include *every distinct point that is present in the abstract*, expanded and explained.\n"
            "- When the abstract hints at a component (e.g., 'framework', 'benchmark', 'pipeline'), explain what that likely means in practice, "
            "but clearly label any assumptions.\n\n"
            "Output format (Markdown) MUST include these sections:\n"
            "1) Paper identity (title, topic, authors, arxiv id, pdf url)\n"
            "2) 5-line 'What this paper is trying to do'\n"
            "3) Background for beginners (define required concepts)\n"
            "4) Problem statement (what’s broken / hard)\n"
            "5) Core idea (the main insight)\n"
            "6) Method / system (components + how they interact)\n"
            "7) Training / data (what data is needed; if missing say so)\n"
            "8) Evaluation (benchmarks/metrics/baselines; if missing say so)\n"
            "9) Claims & evidence (list each claim; what evidence would support it)\n"
            "10) Practical implementation notes (how you’d build it; pseudocode if possible)\n"
            "11) Limitations / failure modes (from abstract + obvious ones)\n"
            "12) Glossary (simple definitions)\n"
            "13) 'If you now read the PDF' guide (what to look for section-by-section)\n"
            "14) 10 self-test questions + short answers\n\n"
            f"Title: {paper['title']}\n\n"
            f"Authors: {', '.join(paper.get('authors', []))}\n\n"
            f"Abstract: {paper['abstract']}\n\n"
            f"arXiv id: {paper.get('arxiv_id','')}\n"
            f"PDF: {paper.get('pdf_url','')}\n"
        )
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = model.invoke([system, human])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def answer(paper: Dict[str, Any], notes_md: str, question: str) -> str:
    """
    Answers the user's question using the saved notes + abstract.
    """

    system = SystemMessage(
        content=(
            "You are an expert academic assistant.\n"
            "Answer using ONLY the provided notes + abstract.\n"
            "If the answer is not contained there, say 'Not specified in the provided notes/abstract' "
            "and suggest what section of the PDF to check.\n"
            "Be simple, direct, and correct.\n"
        )
    )

    human = HumanMessage(
        content=(
            f"Paper title: {paper['title']}\n"
            f"Abstract: {paper['abstract']}\n\n"
            f"Notes (Markdown):\n{notes_md}\n\n"
            f"Question: {question}\n\n"
            "Answer in Markdown. If helpful, include a short bullet list and a tiny example.\n"
        )
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = model.invoke([system, human])
    return resp.content if isinstance(resp.content, str) else str(resp.content)

