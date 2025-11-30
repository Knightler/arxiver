# src/state.py
# type: ignore
from __future__ import annotations

from typing import Any, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from src.graph import fetch, explain


class State(TypedDict, total=False):
    topic: str
    paper: Dict[str, Any]
    notes_md: str


app = StateGraph(State)

app.add_node("fetch", lambda state: {"paper": fetch(state["topic"])})
app.add_node("explain", lambda state: {"notes_md": explain(state["paper"])})

app.add_edge(START, "fetch")
app.add_edge("fetch", "explain")
app.add_edge("explain", END)

graph = app.compile()
