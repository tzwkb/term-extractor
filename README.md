# Term Extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

English | [中文](README_ZH.md)

LLM-powered bilingual terminology extraction prototype for articles and documents.

## Features

- **Text Splitting** — Intelligent chunking for large documents
- **LLM Processing** — Leverages OpenAI API for accurate term recognition
- **Batch Processing** — Handle multiple files in one run
- **Multi-format Input** — Supports PDF, DOCX, and plain text
- **Configurable Prompts** — Customizable system and user prompts via config.py

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set your OpenAI API key:

```python
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

Or set via environment variable:

```bash
export OPENAI_API_KEY="your-key"
```

## Usage

```bash
python main.py --file input.txt --format json
python main.py --file input.pdf --format csv --bilingual
python main.py --monolingual --model gpt-4o
```

## Project Structure

```
.
├── main.py           # Entry point
├── llm_processor.py  # Core LLM interaction logic
├── config.py         # API keys and prompts
├── file_processor.py # Input file handling
└── text_splitter.py  # Document chunking utility
```

## License

[MIT](LICENSE)
