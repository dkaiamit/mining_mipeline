# src/ner/annotation_converter.py
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class AnnotationConverter:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path

    def load_annotations(self) -> List[Dict]:
        try:
            with open(self.annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} annotated records.")
            return data
        except Exception as e:
            logger.error("Failed to load annotations.", exc_info=True)
            raise e

    def extract_project_entities(self, data: List[Dict]) -> List[Tuple[str, List[Tuple[str, str]]]]:
        from nltk.tokenize import word_tokenize

        dataset = []
        for item in data:
            text = item['data']['text']
            labels = item['annotations'][0]['result']
            label_map = {i: 'O' for i in range(len(text))}

            for entity in labels:
                start = entity['value']['start']
                end = entity['value']['end']
                label = entity['value']['labels'][0]

                if label == "PROJECT":
                    for i in range(start, end):
                        if i == start:
                            label_map[i] = 'B-PROJECT'
                        else:
                            label_map[i] = 'I-PROJECT'

            # Convert characters to tokens and labels
            tokens = word_tokenize(text)
            token_labels = []
            idx = 0

            for token in tokens:
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                token_label = label_map.get(idx, 'O')
                token_labels.append((token, token_label))
                idx += len(token)

            dataset.append((text, token_labels))

        logger.info(f"Prepared {len(dataset)} samples with token-level labels.")
        return dataset
