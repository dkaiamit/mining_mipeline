# Mining Project Intelligence Pipeline

## Overview

This project implements an automated pipeline to identify mining project names from unstructured geological and mining reports (PDF files) and infer their approximate geographic coordinates. The goal is to build a robust, modular system that can handle noisy, real-world documents and provide structured outputs suitable for further analysis.

## Problem Statement

Given a set of mining and geological PDF reports, the pipeline must:

1. Extract mining project names accurately using a Named Entity Recognition (NER) approach.
2. Infer geographic coordinates (latitude and longitude) for each project mention based on contextual text cues.

## Approach and Design Decisions

### PDF Text Extraction

We used PyMuPDF (`fitz`) for extracting text from PDFs because it is fast, handles multi-page documents well, and provides reliable text output even with varying layouts.

### NER Model Selection

We tested both `bert-base-cased` and `roberta-base` models from Hugging Face's Transformers library. Initially, `bert-base-cased` provided inconsistent results with noisy and repeated predictions. We switched to `roberta-base` because it generally performs better in NER tasks on noisy, informal text, as shown in various benchmarks.

We fine-tuned `roberta-base` on the provided annotation data, converting the raw JSON annotations to token-label pairs in the IOB scheme (`B-PROJECT`, `I-PROJECT`, `O`). We chose Hugging Face's Trainer API because it simplifies training, evaluation, and saving models, while supporting GPU acceleration if available.

### Geolocation Inference

The second phase of the pipeline attempts to infer geographic coordinates of identified project mentions.

We adopted a hybrid approach:

- **Heuristic/Database Lookup**: For known project names, we return coordinates from a small lookup table.
- **LLM-based Fallback**: If the project is not found in the lookup, we query Gemini 2.0 Flash (from Google AI Studio) to extract likely coordinates based on the context snippet around the project mention.

Due to API limitations in free-tier LLM accounts, the fallback inference may return "Unknown" or incomplete results. However, the pipeline is modular and ready for use with higher-capacity APIs or improved heuristic databases in a real production environment.

## Project Structure


```
mining_project_pipeline/
│
├── data/
│ ├── annotations.json
│ └── pdfs/
│
├── output/
│ ├── ner_model/
│ ├── ner_predictions.jsonl
│ └── ner_predictions_with_geo.jsonl
│
├── src/
│ └── ner/
│ ├── annotation_converter.py
│ ├── dataset_preparer.py
│ ├── ner_model.py
│ ├── infer_ner.py
│ ├── pdf_reader.py
│ ├── utils.py
│ └── geolocation_infer.py
│
├── train_ner.py
├── run_inference.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create a Python virtual environment:
```
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your Gemini API key:

- Create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## How to Run

1. Train the NER model:
```
python train_ner.py roberta-base
```
2. Run the NER inference:

```
python run_inference.py
```
3. Run geolocation inference:
```
python src/ner/geolocation_infer.py
```
## Libraries Used

- transformers
- torch
- datasets
- PyMuPDF
- python-dotenv
- google-generativeai

## Notes and Future Work

- Due to the free-tier limitations of Gemini, geolocation inference results may be incomplete. In practice, this component can be swapped with more robust services or expanded heuristic databases.
- The NER model can be further improved with additional domain-specific training data.
- Containerization using Docker is planned to make the pipeline reproducible across systems.
- We can also use Airflow to schedule the entire process. Airflow can help manage dependencies between tasks and provide better monitoring and orchestration of the pipeline. 