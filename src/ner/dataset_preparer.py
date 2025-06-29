# src/ner/dataset_preparer.py
from datasets import Dataset
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self, tokenizer, label2id: Dict[str, int]):
        self.tokenizer = tokenizer
        self.label2id = label2id

    def encode_examples(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        labels = []
        for i, label_seq in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label_seq[word_idx]])
                else:
                    label = label_seq[word_idx]
                    if label.startswith("B-"):
                        label = label.replace("B-", "I-")
                    label_ids.append(self.label2id[label])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def build_dataset(self, token_label_pairs: List[Tuple[str, List[Tuple[str, str]]]]) -> Dataset:
        data = {
            "tokens": [],
            "labels": []
        }

        for _, token_label_list in token_label_pairs:
            tokens = [tok for tok, _ in token_label_list]
            labels = [lbl for _, lbl in token_label_list]
            data["tokens"].append(tokens)
            data["labels"].append(labels)

        logger.info("Converting token-label pairs to HuggingFace Dataset.")
        return Dataset.from_dict(data)
