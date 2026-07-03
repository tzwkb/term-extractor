# Term Extractor

<!-- bilingual-readme:start -->

## 双语说明 / Bilingual Documentation

> 本节提供整篇 README 的中英双语维护说明；下方保留原始详细说明、命令、路径和配置示例。
> This section provides bilingual maintenance notes for the full README; the original detailed notes, commands, paths, and configuration examples are preserved below.

### 中文

**概览**：本地双语术语提取原型，用 LLM 从文章和文档中抽取术语。

**主要能力**：
- 面向文章/文档术语抽取。
- 支持双语术语处理。
- 适合作为本地实验和工具化原型。

**使用方式**：按下方 Python 环境和脚本说明准备输入文件后运行。

**状态**：该仓库仍按当前 README 的说明维护或使用。

**注意事项**：术语结果需人工复核，不应直接视为最终术语库。

### English

**Overview**: Local bilingual terminology extraction prototype using an LLM to extract terms from articles and documents.

**Key capabilities**:
- Targets terminology extraction from articles/documents.
- Supports bilingual term processing.
- Useful as a local experiment and tooling prototype.

**Usage**: Prepare input files and run the scripts according to the Python environment notes below.

**Status**: This repository is maintained or used according to the current README notes.

**Notes**: Extracted terms should be reviewed manually before being treated as a final glossary.

<!-- bilingual-readme:end -->

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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