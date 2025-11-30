# src/main.py
# type: ignore
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

from src.state import graph

def main():
    result = graph.invoke({
        "topic": "reinforcement learning autonomous driving",
        "questions": [
            "What are the main challenges in applying reinforcement learning to autonomous driving?",
            "Can you summarize the key findings of the survey?"
        ]
    })

    print("Final State:")
    print(result)

if __name__ == "__main__":
    main()
