import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from openai import OpenAI
    import tiktoken
except ImportError:
    print("Install required: pip install openai tiktoken")
    raise


class LLMProcessor:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", base_dir: str = "batch_results"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=3)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self._setup_logging()
        self.semaphore = threading.Semaphore(5)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.base_dir / "llm_processor.log", encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        try:
            return len(tiktoken.encoding_for_model(model).encode(text))
        except Exception:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def process_single_text(self, text: str, custom_id: str, system_prompt: str,
                            user_prompt_template: str, model: str = "gpt-4-turbo-preview",
                            temperature: float = 0.1, max_tokens: int = 4096,
                            source_file: str = None) -> Dict[str, Any]:
        with self.semaphore:
            try:
                from config import get_token_param
                user_prompt = user_prompt_template.format(text=text)
                api_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    get_token_param(model): max_tokens
                }
                response = self.client.chat.completions.create(**api_params)
                return self._process_response(response, custom_id, model, source_file)
            except Exception as e:
                self.logger.error(f"Failed {custom_id}: {e}")
                return self._error_result(custom_id, model, source_file, str(e))

    def _process_response(self, response, custom_id: str, model: str, source_file: str) -> Dict[str, Any]:
        try:
            content = response.choices[0].message.content
            usage_info = {
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
            }
            return {
                "custom_id": custom_id,
                "extracted_terms": self._parse_json(content),
                "usage": usage_info,
                "model": response.model,
                "source_file": source_file,
                "created": int(time.time())
            }
        except Exception as e:
            return self._error_result(custom_id, model, source_file, str(e))

    def _parse_json(self, content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if not isinstance(parsed, dict):
                return {"raw_content": str(parsed)}
            if "terms" not in parsed:
                if len(parsed) == 1:
                    k = list(parsed.keys())[0]
                    if isinstance(parsed[k], list):
                        return {"terms": parsed[k]}
                return {"raw_content": str(parsed)}
            return parsed
        except json.JSONDecodeError:
            try:
                start = content.find('{')
                if start == -1:
                    start = content.find('[')
                if start != -1:
                    end = content.rfind('}') if content[start] == '{' else content.rfind(']')
                    if end > start:
                        return json.loads(content[start:end+1].strip())
            except Exception:
                pass
            return {"raw_content": content}

    def _error_result(self, custom_id: str, model: str, source_file: str, error_msg: str) -> Dict[str, Any]:
        return {
            "custom_id": custom_id,
            "error": error_msg,
            "extracted_terms": {"raw_content": f"Failed: {error_msg}"},
            "usage": {"total_tokens": 0},
            "model": model,
            "source_file": source_file,
            "created": int(time.time())
        }

    def process_batch_concurrent(self, texts: List[str], system_prompt: str,
                                 user_prompt_template: str, model: str = "gpt-4-turbo-preview",
                                 temperature: float = 0.1, max_tokens: int = 4096,
                                 max_concurrent: int = 10, source_files: List[str] = None) -> List[Dict[str, Any]]:
        self.semaphore = threading.Semaphore(max_concurrent)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for i, text in enumerate(texts):
                custom_id = f"term-extraction-{i+1}"
                sf = source_files[i] if source_files and i < len(source_files) else None
                future = executor.submit(self.process_single_text, text=text, custom_id=custom_id,
                                         system_prompt=system_prompt, user_prompt_template=user_prompt_template,
                                         model=model, temperature=temperature, max_tokens=max_tokens, source_file=sf)
                futures[future] = custom_id

            results = []
            completed = 0
            for future in as_completed(futures):
                custom_id = futures[future]
                try:
                    results.append(future.result())
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{len(texts)}")
                except Exception as e:
                    idx = int(custom_id.split('-')[-1]) - 1
                    sf = source_files[idx] if source_files and idx < len(source_files) else f"text_{idx+1}.txt"
                    results.append(self._error_result(custom_id, model, sf, str(e)))

        results.sort(key=lambda x: x.get("custom_id", ""))
        return results

    def deduplicate_terms(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_terms = {}
        for result in results:
            extracted = result.get("extracted_terms", {})
            if "terms" not in extracted or not isinstance(extracted["terms"], list):
                continue
            for term in extracted["terms"]:
                if not isinstance(term, dict):
                    continue
                eng = term.get("eng_term", "").strip()
                zh = term.get("zh_term", "").strip()
                if not eng and not zh and "term" in term:
                    key = term["term"].strip().lower()
                    all_terms.setdefault(key, []).append({
                        "original_term": term["term"],
                        "source_file": result.get("source_file", "")
                    })
                else:
                    key = f"{eng.lower()}|{zh}"
                    all_terms.setdefault(key, []).append({
                        "original_eng_term": eng,
                        "original_zh_term": zh,
                        "source_file": result.get("source_file", "")
                    })

        merged_terms = []
        duplicate_count = 0
        for term_list in all_terms.values():
            if len(term_list) > 1:
                duplicate_count += len(term_list) - 1
            merged_terms.append(self._merge_term(term_list))

        return [{
            "custom_id": "merged_terms",
            "extracted_terms": {
                "terms": merged_terms,
                "total_terms": len(merged_terms),
                "duplicates_removed": duplicate_count
            },
            "usage": {"total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in results)},
            "model": "merged",
            "created": max((r.get("created", 0) for r in results), default=0)
        }]

    def _merge_term(self, term_list: List[Dict]) -> Dict:
        sources = list(set(t.get("source_file", "") for t in term_list if t.get("source_file")))
        if "original_eng_term" in term_list[0]:
            best_eng = next((t["original_eng_term"] for t in term_list if t["original_eng_term"] and t["original_eng_term"][0].isupper()), term_list[0]["original_eng_term"])
            result = {"eng_term": best_eng, "zh_term": term_list[0]["original_zh_term"]}
        else:
            best_term = next((t["original_term"] for t in term_list if t["original_term"] and t["original_term"][0].isupper()), term_list[0]["original_term"])
            result = {"term": best_term}
        result["source_files" if len(sources) > 1 else "source_file"] = sources if len(sources) > 1 else (sources[0] if sources else "")
        return result

    def _extract_source_filename(self, source_files: List[str]) -> str:
        if not source_files:
            return "unknown"
        import re
        clean = re.sub(r'\s*-\s*fragment\s*\d+/\d+\s*\([^)]+\)', '', source_files[0])
        filename = Path(clean.strip()).stem
        return re.sub(r'[^\w\u4e00-\u9fff]', '_', filename)[:50]

    def _count_total_terms(self, results: List[Dict[str, Any]]) -> int:
        for result in results:
            if result.get("custom_id") == "merged_terms":
                terms = result.get("extracted_terms", {})
                return len(terms.get("terms", [])) or terms.get("total_terms", 0)
        return 0

    def save_processed_results(self, results: List[Dict[str, Any]], output_format: str = "json",
                               source_filename: str = "", model_name: str = "", total_terms: int = 0) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if source_filename and model_name and total_terms > 0:
            base = f"{source_filename}_{timestamp}_{model_name}_{total_terms}terms"
        else:
            base = f"processed_terms_{timestamp}"

        if output_format == "json":
            return self._save_json(results, base)
        elif output_format == "csv":
            return self._save_csv(results, base)
        elif output_format == "txt":
            return self._save_txt(results, base)
        elif output_format == "excel":
            return self._save_excel(results, base)
        elif output_format == "tbx":
            return self._save_tbx(results, base)
        raise ValueError(f"Unsupported format: {output_format}")

    def _save_json(self, results, base: str) -> str:
        out = self.base_dir / f"{base}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return str(out)

    def _save_csv(self, results, base: str) -> str:
        import csv
        out = self.base_dir / f"{base}.csv"
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["custom_id", "eng_term", "zh_term", "source_file", "model", "tokens"])
            for result in results:
                terms = result.get("extracted_terms", {})
                if "terms" in terms and isinstance(terms["terms"], list):
                    for term in terms["terms"]:
                        eng = term.get("eng_term", "")
                        zh = term.get("zh_term", "")
                        if not eng and not zh:
                            single = term.get("term", "")
                            if any('\u4e00' <= c <= '\u9fff' for c in single):
                                zh = single
                            else:
                                eng = single
                        writer.writerow([result.get("custom_id", ""), eng, zh,
                                         term.get("source_file", result.get("source_file", "")),
                                         result.get("model", ""), result.get("usage", {}).get("total_tokens", 0)])
                else:
                    writer.writerow([result.get("custom_id", ""), "", str(terms.get("raw_content", "")),
                                     result.get("source_file", ""), result.get("model", ""),
                                     result.get("usage", {}).get("total_tokens", 0)])
        return str(out)

    def _save_txt(self, results, base: str) -> str:
        out = self.base_dir / f"{base}.txt"
        with open(out, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"=== Result {i}: {result.get('custom_id', 'unknown')} ===\n")
                terms = result.get("extracted_terms", {})
                if "terms" in terms and isinstance(terms["terms"], list):
                    f.write(f"Terms: {len(terms['terms'])}\n")
                    f.write(f"Source: {result.get('source_file', 'unknown')}\n")
                    f.write(f"Model: {result.get('model', 'unknown')}\n")
                    f.write(f"Tokens: {result.get('usage', {}).get('total_tokens', 0)}\n\n")
                    for j, term in enumerate(terms["terms"], 1):
                        eng = term.get("eng_term", "")
                        zh = term.get("zh_term", "")
                        if not eng and not zh:
                            f.write(f"{j}. {term.get('term', '')}\n")
                        else:
                            f.write(f"{j}. {eng}\n   {zh}\n")
                        if term.get('source_file'):
                            f.write(f"   Source: {term['source_file']}\n")
                else:
                    f.write(f"Raw: {terms.get('raw_content', 'none')}\n")
                f.write("=" * 50 + "\n\n")
        return str(out)

    def _save_excel(self, results, base: str) -> str:
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            raise ImportError("Install openpyxl: pip install openpyxl")

        out = self.base_dir / f"{base}.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Terms"

        hf = Font(bold=True, color="FFFFFF")
        hfill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        ha = Alignment(horizontal="center", vertical="center")
        border = Border(left=Side(style="thin"), right=Side(style="thin"),
                        top=Side(style="thin"), bottom=Side(style="thin"))

        headers = ["#", "eng_term", "zh_term", "source_file", "model", "tokens", "created"]
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = hf; cell.fill = hfill; cell.alignment = ha; cell.border = border

        row_num = 2
        term_count = 0
        for result in results:
            terms = result.get("extracted_terms", {})
            if "terms" in terms and isinstance(terms["terms"], list):
                for term in terms["terms"]:
                    term_count += 1
                    sf = ""
                    if term.get("source_files"):
                        sf = "; ".join(term["source_files"]) if isinstance(term["source_files"], list) else str(term["source_files"])
                    elif term.get("source_file"):
                        sf = term["source_file"]
                    elif result.get("source_file"):
                        sf = result["source_file"]

                    created = ""
                    if result.get("created"):
                        created = datetime.fromtimestamp(result["created"]).strftime("%Y-%m-%d %H:%M:%S")

                    eng = term.get("eng_term", "")
                    zh = term.get("zh_term", "")
                    if not eng and not zh:
                        single = term.get("term", "")
                        if any('\u4e00' <= c <= '\u9fff' for c in single):
                            zh = single
                        else:
                            eng = single

                    for col, val in enumerate([term_count, eng, zh, sf, result.get("model", ""),
                                               result.get("usage", {}).get("total_tokens", 0), created], 1):
                        cell = ws.cell(row=row_num, column=col, value=val)
                        cell.border = border
                        if col in [1, 6]:
                            cell.alignment = Alignment(horizontal="center")
                    row_num += 1

        for col in range(1, len(headers) + 1):
            cl = get_column_letter(col)
            max_len = max((len(str(row.value)) for row in ws[cl] if row.value), default=0)
            ws.column_dimensions[cl].width = min(max(max_len + 2, 10), 50)

        stats_ws = wb.create_sheet("Stats")
        for ri, (k, v) in enumerate([("Total Terms", term_count), ("Results", len(results)),
                                     ("Total Tokens", sum(r.get("usage", {}).get("total_tokens", 0) for r in results)),
                                     ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))], 1):
            for ci, val in enumerate([k, v], 1):
                cell = stats_ws.cell(row=ri, column=ci, value=val)
                if ri == 1:
                    cell.font = hf; cell.fill = hfill; cell.alignment = ha
                cell.border = border

        wb.save(out)
        return str(out)

    def _save_tbx(self, results, base: str) -> str:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        import re

        out = self.base_dir / f"{base}.tbx"
        root = ET.Element("tbx", attrib={"type": "TBX-Default", "style": "dct",
                                          "xml:lang": "en", "xmlns": "urn:iso:std:iso:30042:ed-2"})
        header = ET.SubElement(root, "tbxHeader")
        file_desc = ET.SubElement(header, "fileDesc")
        title = ET.SubElement(ET.SubElement(file_desc, "titleStmt"), "title")
        title.text = "Term Base"
        pub = ET.SubElement(ET.SubElement(file_desc, "publicationStmt"), "p")
        pub.text = f"Generated by LLM Term Extractor on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        body = ET.SubElement(ET.SubElement(root, "text"), "body")
        count = 0
        for result in results:
            terms = result.get("extracted_terms", {})
            if "terms" not in terms or not isinstance(terms["terms"], list):
                continue
            for term in terms["terms"]:
                count += 1
                entry = ET.SubElement(body, "termEntry", attrib={"id": f"term_{count}"})
                admin_grp = ET.SubElement(entry, "adminGrp")
                ET.SubElement(admin_grp, "admin", attrib={"type": "subjectField"}).text = "general"

                sf = ""
                if term.get("source_files"):
                    sf = "; ".join(term["source_files"]) if isinstance(term["source_files"], list) else str(term["source_files"])
                elif term.get("source_file"):
                    sf = term["source_file"]
                if sf:
                    ET.SubElement(admin_grp, "admin", attrib={"type": "source"}).text = sf

                eng = term.get("eng_term", "")
                zh = term.get("zh_term", "")
                if not eng and not zh:
                    single = term.get("term", "")
                    lang = "zh" if re.search(r'[\u4e00-\u9fff]', single) else "en"
                    lg = ET.SubElement(entry, "langGrp", attrib={"xml:lang": lang})
                    ET.SubElement(ET.SubElement(lg, "termGrp"), "term").text = single
                else:
                    if eng:
                        lg = ET.SubElement(entry, "langGrp", attrib={"xml:lang": "en"})
                        ET.SubElement(ET.SubElement(lg, "termGrp"), "term").text = eng
                    if zh:
                        lg = ET.SubElement(entry, "langGrp", attrib={"xml:lang": "zh"})
                        ET.SubElement(ET.SubElement(lg, "termGrp"), "term").text = zh

        rough = ET.tostring(root, encoding='unicode')
        pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding=None)
        lines = [l for l in pretty.split('\n') if l.strip()]
        final = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE tbx SYSTEM "TBXcoreStructV02.dtd">\n' + '\n'.join(lines[1:])
        with open(out, 'w', encoding='utf-8') as f:
            f.write(final)
        return str(out)

    def run_extraction_only(self, texts: List[str], system_prompt: str, user_prompt_template: str,
                            model: str = "gpt-4-turbo-preview", temperature: float = 0.1,
                            max_tokens: int = 4096, max_concurrent: int = 10,
                            description: str = "term extraction", source_files: List[str] = None) -> Dict[str, Any]:
        results = self.process_batch_concurrent(texts=texts, system_prompt=system_prompt,
                                                user_prompt_template=user_prompt_template, model=model,
                                                temperature=temperature, max_tokens=max_tokens,
                                                max_concurrent=max_concurrent, source_files=source_files)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self.base_dir / f"raw_results_{timestamp}.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        merged = self.deduplicate_terms(results)
        return {"raw_results": results, "merged_results": merged, "raw_file": str(raw_file)}


def _save_intermediate_text(file_path: str, text: str):
    out_dir = Path("extracted_texts")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{Path(file_path).stem}.txt"
    try:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Warning: could not save intermediate text: {e}")


def load_texts_from_file(file_path: str, chunk_size: Optional[int] = None,
                         use_smart_splitter: bool = True, overlap_size: int = 200) -> List[str]:
    from file_processor import FileProcessor
    from text_splitter import TextSplitter

    processor = FileProcessor()
    if file_path.endswith('.pdf'):
        texts = processor.extract_pdf_text(file_path)
    elif file_path.endswith(('.docx', '.doc')):
        texts = processor.extract_docx_text(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [f.read()]

    if not texts or not any(t.strip() for t in texts):
        raise ValueError("Empty file or failed to extract content")

    full_text = '\n\n'.join(texts)
    _save_intermediate_text(file_path, full_text)

    if chunk_size and use_smart_splitter:
        splitter = TextSplitter(max_tokens=max(chunk_size // 4, 500), overlap_tokens=min(overlap_size // 4, chunk_size // 40))
        return splitter.split_text_with_metadata(full_text, Path(file_path).name)
    elif chunk_size:
        splitter = TextSplitter(max_tokens=10000)
        chunks = splitter.split_by_paragraphs(full_text)
        name = Path(file_path).name
        return [f"[File: {name} - fragment {i}/{len(chunks)}]\n{c}" if len(chunks) > 1 else f"[File: {name}]\n{c}"
                for i, c in enumerate(chunks, 1)]
    else:
        return [f"[File: {Path(file_path).name}]\n{full_text}"]
