# src/state.py
# type: ignore
"""
State management for the Arxiver application.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph
from graph import FetchNode, ExplainNode, QANode

# Define the state structure
class State(TypedDict, total=False):
    paper_title: str                   # input / during FetchNode
    paper: Dict[str, Any]              # after FetchNode
    explanations: Dict[str, str]       # after ExplainNode
    questions: List[str]               # input / during Q&A
    answers: Dict[str, str]            # after QANode

# Build the state graph
app = StateGraph(State)

# Define nodes
fetchnode = FetchNode()
app.add_node("fetch",
             lambda state: {'paper': fetchnode.fetch(state["paper_title"])})

explainnode = ExplainNode()
app.add_node("explain",
             lambda state: {'explanations': explainnode.explain(state["paper"])})

qanode = QANode()
app.add_node("qa",
             lambda state: {
                 'answers': {
                     q: qanode.answer(state["paper"], q)
                      for q in state["questions"]
                 }})

# Define edges
app.add_edge("fetch", "explain")
app.add_edge("explain", "qa")
app.add_edge("qa", "END")
app.add_edge("START", "fetch")

graph = app.compile()