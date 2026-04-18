# Changelog

## [Unreleased] - 2026-04-18

### Changed

- **config.py**: Stripped to essentials — removed `OUTPUT_FORMATS`, `TERM_PROCESSING`, `FILE_PROCESSING`, `LOGGING_CONFIG` dicts and all helper functions except `get_token_param`. Renamed `get_token_param_name` → `get_token_param`.
- **text_splitter.py**: Reduced from ~675 to ~195 lines. Removed convenience wrappers and redundant split strategies. All labels converted from Chinese to English.
- **file_processor.py**: Reduced from ~517 to ~141 lines. Removed OCR/image support, utility functions, and broken `extract_with_custom_params`. `FileProcessor` now exposes only `extract_pdf_text` and `extract_docx_text`.
- **llm_processor.py**: Reduced from ~1385 to ~496 lines. Removed `run_complete_pipeline`, merged deduplication helpers into `deduplicate_terms`, fixed `LOGGING_CONFIG` import (removed), fixed `get_token_param_name` → `get_token_param`. All Chinese strings converted to English.
- **main.py**: Rewritten from ~735 to ~90 lines. Replaced interactive menu system with a minimal argparse CLI. All Chinese text removed.

### Removed

- Interactive multi-step menus for file selection, splitting config, and output format
- Sample Chinese aerospace texts
- Logging config abstraction (`LOGGING_CONFIG`)
- OCR and image extraction support
- `run_complete_pipeline` method

### Fixed

- `llm_processor.py` importing removed `LOGGING_CONFIG` from `config.py`
- `llm_processor.py` calling renamed `get_token_param_name` (now `get_token_param`)
