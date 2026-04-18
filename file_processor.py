import os
import logging
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class DependencyManager:
    def __init__(self):
        self.available_modules = {}
        self._check_dependencies()

    def _check_dependencies(self):
        if importlib.util.find_spec('pdfminer'):
            self.available_modules['pdf'] = ['pdfminer.six']
        elif importlib.util.find_spec('PyPDF2'):
            self.available_modules['pdf'] = ['PyPDF2']
        else:
            self.available_modules['pdf'] = []

        if importlib.util.find_spec('docx'):
            self.available_modules['docx'] = ['python-docx']
        else:
            self.available_modules['docx'] = []

    def is_available(self, module: str) -> bool:
        return bool(self.available_modules.get(module, []))


deps = DependencyManager()


class FileTypeDetector:
    EXTENSION_MAP = {
        '.txt': ('text', 'text/plain'),
        '.md': ('text', 'text/markdown'),
        '.html': ('text', 'text/html'),
        '.xml': ('text', 'application/xml'),
        '.pdf': ('pdf', 'application/pdf'),
        '.docx': ('docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        '.doc': ('doc', 'application/msword'),
    }

    @staticmethod
    def detect_file_type(file_path: str) -> Tuple[str, str]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = Path(file_path).suffix.lower()
        return FileTypeDetector.EXTENSION_MAP.get(ext, ('unknown', 'application/octet-stream'))


class PlainTextExtractor:
    def extract(self, file_path: str) -> List[str]:
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if content:
                        return [content]
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file {file_path}")


class PDFExtractor:
    def __init__(self, dependencies: DependencyManager):
        self.deps = dependencies

    def extract(self, file_path: str) -> List[str]:
        if not deps.is_available('pdf'):
            raise RuntimeError("PDF library not available, install pdfminer.six")

        if 'pdfminer.six' in deps.available_modules['pdf']:
            return self._extract_with_pdfminer(file_path)
        return self._extract_with_pypdf2(file_path)

    def _extract_with_pdfminer(self, file_path: str) -> List[str]:
        import re
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5,
                            word_margin=0.3, boxes_flow=0.5, detect_vertical=True, all_texts=True)
        full_text = extract_text(file_path, laparams=laparams)

        if not full_text or len(full_text.strip()) < 50:
            raise ValueError(f"PDF text extraction failed or too short: {Path(file_path).name}")

        full_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_text)
        full_text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', full_text)
        full_text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', full_text)
        full_text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', full_text)
        full_text = re.sub(r'\s+', ' ', full_text)
        return [full_text.strip()]

    def _extract_with_pypdf2(self, file_path: str) -> List[str]:
        import re, PyPDF2
        texts = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
                    text = re.sub(r'\s+', ' ', text)
                    texts.append(text.strip())
        if not texts:
            raise ValueError("PDF is empty or unreadable")
        return texts


class DOCXExtractor:
    def extract(self, file_path: str) -> List[str]:
        if not deps.is_available('docx'):
            raise RuntimeError("DOCX library not available, install python-docx")
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise ValueError("DOCX file is empty")
        return ['\n\n'.join(paragraphs)]


class FileProcessor:
    def __init__(self):
        self.deps = DependencyManager()
        self.extractors = {'text': PlainTextExtractor()}
        if self.deps.is_available('pdf'):
            self.extractors['pdf'] = PDFExtractor(self.deps)
        if self.deps.is_available('docx'):
            self.extractors['docx'] = DOCXExtractor()
            self.extractors['doc'] = DOCXExtractor()

    def extract_pdf_text(self, file_path: str) -> List[str]:
        return self.extractors['pdf'].extract(file_path)

    def extract_docx_text(self, file_path: str) -> List[str]:
        return self.extractors['docx'].extract(file_path)
