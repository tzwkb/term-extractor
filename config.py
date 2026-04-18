import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = "gpt-4o"

MODEL_NEW = ["gpt-5", "2025", "gpt-4o-2024-08-06"]

def get_token_param(model: str) -> str:
    return "max_completion_tokens" if any(m in model.lower() for m in MODEL_NEW) else "max_tokens"

BATCH_CONFIG = {
    "temperature": 0,
    "max_output_tokens": 128000,
    "max_concurrent": 10,
}

TEXT_SPLITTING = {
    "default_chunk_size": 12000,
    "default_overlap_size": 800,
    "min_chunk_size": 1000,
    "max_chunk_size": 400000,
    "max_overlap_ratio": 0.1,
    "whole_document_threshold": 300000,
}

SYSTEM_PROMPT = """You are a precise keyword extraction system. Extract ONLY the bilingual term pairs from the "关键词" (Chinese keywords) and "Keywords" (English keywords) sections of academic papers. Ignore all other content."""

def get_user_prompt(text: str, bilingual: bool = True) -> str:
    if bilingual:
        return """Your response must contain only a JSON object with no additional text.

Task: Extract ONLY the bilingual term pairs from the keyword sections:
- Find the "关键词" section (Chinese keywords)
- Find the "Keywords" section (English keywords)
- Match corresponding terms between both sections
- Ignore all other content

Output format:
{{
  "terms": [
    {{"eng_term": "English keyword", "zh_term": "Chinese keyword"}}
  ]
}}

Text to analyze:
{text}
"""
    else:
        return """Your response must contain only a JSON object with no additional text.

Task: Extract ONLY terms from the "Keywords" or "关键词" section. Ignore all other content.

Output format:
{{
  "terms": [
    {{"term": "keyword1"}}
  ]
}}

Text to analyze:
{text}
"""
