# src/ner/pdf_reader.py

import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_pages(self):
        try:
            doc = fitz.open(self.pdf_path)
            pages = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                pages.append((page_num, text))
            logger.info(f"Extracted {len(pages)} pages from {self.pdf_path}")
            return pages
        except Exception as e:
            logger.error(f"Failed to extract text from {self.pdf_path}", exc_info=True)
            raise e
