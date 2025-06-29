# train_ner.py

import sys
from src.ner.annotation_converter import AnnotationConverter
from src.ner.dataset_preparer import DatasetPreparer
from src.ner.ner_model import NERModel
from src.ner.utils import init_logging

def main():
    init_logging()

    if len(sys.argv) < 2:
        print("Usage: python train_ner.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"\n🚀 Starting training with model: {model_name}")

    converter = AnnotationConverter("data/annotations.json")
    annotated_data = converter.load_annotations()
    token_label_pairs = converter.extract_project_entities(annotated_data)

    ner_model = NERModel(model_name=model_name)
    preparer = DatasetPreparer(ner_model.tokenizer, ner_model.label2id)
    dataset = preparer.build_dataset(token_label_pairs)
    encoded = dataset.map(preparer.encode_examples, batched=True)

    ner_model.train(encoded)

if __name__ == "__main__":
    main()
