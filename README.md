# arxiver

Command-line tool that fetches research papers from arXiv, explains them in
plain language, and optionally answers follow-up questions.

## Installation

```bash
uv sync          # or pip install .
uv tool install .  # optional: make `arxiver` available globally via uv
```

Alternatively, use `pip install .` (or `pipx install .`) inside the project
root to expose the `arxiver` executable.

## Usage

```bash
arxiver -n "attention is all you need"
arxiver -n "diffusion models" -q "What problem does it solve?" -q "Key math?"
```

Run `arxiver -h` to see the full list of supported options.