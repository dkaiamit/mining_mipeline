# src/ner/ner_model.py

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
import torch
import logging
import os

logger = logging.getLogger(__name__)

class NERModel:
    def __init__(self, model_name, label_list=None):
        self.model_name = model_name
        self.label_list = label_list or ["O", "B-PROJECT", "I-PROJECT"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
        add_prefix_space=True if "roberta" in self.model_name.lower() else False
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def train(self, dataset, output_dir_base="./output"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {device}")

        output_dir = os.path.join(output_dir_base, f"{self.model_name.replace('/', '_')}_model")
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            logging_strategy="no",
            save_strategy="no",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"✅ Finished training {self.model_name}. Model saved to {output_dir}")
