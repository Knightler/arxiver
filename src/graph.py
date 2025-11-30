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
from langchain_community.retrievers import ArxivRetriever

import arxiv


from dotenv import load_dotenv


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
logging.info("Loading environment variables from .env file")
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Fetches the paper based on the query.
def fetch(name: str) -> Dict[str, Any]:
  """
  Fetches the paper from the Arxiv API based on the query.

  IO:
    Input: name (str)
    Output: paper (Dict[str, Any])
  """

  logger.info(f"Searching arXiv for: {name}")

  retriever = ArxivRetriever(
     load_max_docs=1,
     get_full_documents=True,
     arxiv_search=arxiv.Search,
     arxiv_exceptions=(arxiv.ArxivError,)
  )

  results = retriever.invoke(name)

  if not results:
      logger.warning("No papers found.")
      raise ValueError("No papers found for the given query.")

  paper = results[0]
  logger.info(f"Fetched paper: {paper.metadata['title']}")

  return {
      "title": paper.metadata["title"],
      "authors": paper.metadata["authors"],
      "abstract": paper.metadata["abstract"],
      "pdf_url": paper.metadata["pdf_url"],
  }


def explain(paper: Dict[str, Any]) -> str:
  """
  Generates a plain-language explanation of the paper's content.

  IO:
    Input: paper (Dict[str, Any])
    Output: explanation (str)
  """

  prompt = [
    SystemMessage(content="You are an expert academic assistant, specializing in explaining research papers in simple terms."),
    HumanMessage(content=f"Explain the following research paper in plain language:\n\nTitle: {paper['title']}\n\nAbstract: {paper['abstract']}")
  ]

  model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  logger.info(f"Generating explanation for paper: {paper['title']}")
  response = model.invoke(prompt)

  content = response.content

  logger.info(f"Explanation generated for paper: {paper['title']}")
  return content if isinstance(content, str) else str(content)



def answer(paper: Dict[str, Any], question: str) -> str:
  """
  Answers a user question based on the paper's content.
  IO:
    Input: paper (Dict[str, Any]), question (str)
    Output: answer (str)

  Args:
    paper (Dict[str, Any]): The paper details including title and abstract.
    question (str): The user's question about the paper.
  """

  prompt = [
      SystemMessage(content="You are an expert academic assistant, specializing in answering questions about research papers."),
      HumanMessage(
          content=(
              f"Based on the following research paper, answer the question in simple language:\n\n"
              f"Title: {paper['title']}\n\n"
              f"Abstract: {paper['abstract']}\n\n"
              f"Question: {question}"
          )
      ),
  ]

  model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  logger.info(f"Answering question for paper: {paper['title']}")
  response = model.invoke(prompt)

  content = response.content
  logger.info(f"Answer generated for paper: {paper['title']}")
  return content if isinstance(content, str) else str(content)

