# run_inference.py

import os
import json
from src.ner.pdf_reader import PDFTextExtractor
from src.ner.infer_ner import NERPredictor
from src.ner.utils import init_logging

def main():
    init_logging("logs/ner_inference.log")
    model_path = "output/roberta-base_model"  # Change if needed
    predictor = NERPredictor(model_path)

    pdf_dir = "data/pdfs"
    output_file = "output/ner_predictions.jsonl"

    with open(output_file, "w", encoding="utf-8") as out:
        for pdf_file in os.listdir(pdf_dir):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            extractor = PDFTextExtractor(os.path.join(pdf_dir, pdf_file))
            pages = extractor.extract_pages()

            for page_number, text in pages:
                results = predictor.extract_project_mentions(text, pdf_file, page_number)
                for record in results:
                    out.write(json.dumps(record) + "\n")

    print(f"Done Inference results saved to: {output_file}")

if __name__ == "__main__":
    main()
