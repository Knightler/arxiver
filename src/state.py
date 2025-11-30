# src/state.py
# type: ignore
"""
State management for the Arxiver application.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
from src.graph import fetch, explain, answer

# Define the state structure
class State(TypedDict, total=False):
    mode: str
    topic: str
    paper: Dict[str, Any]
    explanations: str
    questions: List[str]
    answers: Dict[str, str]

# Build the state graph
app = StateGraph(State)

# Define nodes
app.add_node("fetch",
             lambda state: {'paper': fetch(state["topic"])})

app.add_node("explain",
             lambda state: {'explanations': explain(state["paper"])})

app.add_node("qa",
             lambda state: {
                 'answers': {
                     q: answer(state["paper"], q)
                      for q in state["questions"]
                 }})

# Define edges
app.add_edge(START, "fetch")
app.add_edge("fetch", "explain")
app.add_edge("explain", "qa")
app.add_edge("qa", END)

graph = app.compile()