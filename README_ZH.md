# Term Extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

[English](README.md) | 中文

## 概览

本地双语术语提取原型，用 LLM 从文章和文档中抽取术语。

## 主要能力

- 面向文章/文档术语抽取。
- 支持双语术语处理。
- 适合作为本地实验和工具化原型。

## 使用方式

按下方 Python 环境和脚本说明准备输入文件后运行。

## 注意事项

术语结果需人工复核，不应直接视为最终术语库。

## 命令与配置参考

以下命令、路径和配置键保持原样，复制时请以实际环境为准。

```bash
pip install -r requirements.txt
```

```python
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

```bash
export OPENAI_API_KEY="your-key"
```

```bash
python main.py --file input.txt --format json
python main.py --file input.pdf --format csv --bilingual
python main.py --monolingual --model gpt-4o
```

```
.
├── main.py           # Entry point
├── llm_processor.py  # Core LLM interaction logic
├── config.py         # API keys and prompts
├── file_processor.py # Input file handling
└── text_splitter.py  # Document chunking utility
```
