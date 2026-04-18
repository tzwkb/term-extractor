import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from llm_processor import LLMProcessor, load_texts_from_file
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, BATCH_CONFIG, SYSTEM_PROMPT, get_user_prompt, TEXT_SPLITTING, DEFAULT_MODEL
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def get_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    if cli_key:
        return cli_key
    key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if key:
        return key
    key = input("Enter OpenAI API key: ").strip()
    return key or None


def load_texts(file_path: Optional[str], chunk_size: Optional[int]) -> List[str]:
    if file_path:
        return load_texts_from_file(
            file_path,
            chunk_size=chunk_size or TEXT_SPLITTING["default_chunk_size"],
            use_smart_splitter=True,
            overlap_size=TEXT_SPLITTING["default_overlap_size"]
        )
    print("Enter texts (one per line, blank line to finish):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    if not lines:
        raise ValueError("No input provided")
    return lines


def run(api_key: str, texts: List[str], model: str, output_format: str, bilingual: bool):
    base_url = os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL
    processor = LLMProcessor(api_key=api_key, base_url=base_url)

    source_files = []
    for t in texts:
        import re
        m = re.search(r'\[File: ([^\]]+)\]', t)
        source_files.append(m.group(1) if m else "")

    results = processor.run_extraction_only(
        texts=texts,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=get_user_prompt("{text}", bilingual=bilingual),
        model=model,
        temperature=BATCH_CONFIG["temperature"],
        max_tokens=BATCH_CONFIG["max_output_tokens"],
        max_concurrent=BATCH_CONFIG["max_concurrent"],
        source_files=source_files
    )

    merged = results.get("merged_results", [])
    source_filename = processor._extract_source_filename(source_files)
    model_name = model.replace("-", "").replace(".", "")
    total_terms = processor._count_total_terms(merged)

    out_file = processor.save_processed_results(merged, output_format, source_filename, model_name, total_terms)
    print(f"Done: {total_terms} terms -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM term extractor")
    parser.add_argument("--api-key")
    parser.add_argument("--file")
    parser.add_argument("--format", choices=["json", "csv", "txt", "excel", "tbx"], default="json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--bilingual", action="store_true", default=True)
    parser.add_argument("--monolingual", dest="bilingual", action="store_false")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("No API key provided.")
        sys.exit(1)

    try:
        texts = load_texts(args.file, args.chunk_size)
    except Exception as e:
        print(f"Failed to load texts: {e}")
        sys.exit(1)

    run(api_key, texts, args.model, args.format, args.bilingual)


if __name__ == "__main__":
    main()
