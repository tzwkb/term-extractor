# Term Extractor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

LLM-powered bilingual terminology extraction prototype for articles and documents.

## Features

- **Text Splitting** — Intelligent chunking for large documents
- **LLM Processing** — Leverages OpenAI API for accurate term recognition
- **Batch Processing** — Handle multiple files in one run
- **Multi-format Input** — Supports PDF text extraction and plain text
- **Configurable Prompts** — Customizable system and user prompts via config.py

## Installation

`ash
pip install -r requirements.txt
`

## Configuration

Edit config.py to set your OpenAI API key and adjust batch settings:

`python
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
`

Or set via environment variable:

`ash
export OPENAI_API_KEY="your-key"
`

## Usage

`ash
python main.py
`

Follow the interactive prompts to select input files and extraction options.

## Project Structure

`
.
├── main.py           # Entry point
├── llm_processor.py  # Core LLM interaction logic
├── config.py         # API keys and prompts
├── file_processor.py # Input file handling
└── text_splitter.py  # Document chunking utility
`

## License

[MIT](LICENSE)