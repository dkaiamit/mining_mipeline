# src/ner/infer_ner.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import logging
import re

logger = logging.getLogger(__name__)

class NERPredictor:
    def __init__(self, model_path: str):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

    def get_entities(self, text: str):
        return self.nlp(text)

    def extract_project_mentions(self, text: str, pdf_file: str, page_number: int):
        entities = self.get_entities(text)
        results = []

        for ent in entities:
            if ent['entity_group'] == "PROJECT":
                context = self._get_context(text, ent['start'], ent['end'])
                result = {
                    "pdf_file": pdf_file,
                    "page_number": page_number,
                    "project_name": ent['word'],
                    "context_sentence": context,
                    "coordinates": None
                }
                results.append(result)

        return results

    def _get_context(self, text, start, end, window=150):
        pre = text[max(0, start - window):start]
        mention = text[start:end]
        post = text[end:end + window]
        sentence = f"{pre}{mention}{post}".strip()
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence
