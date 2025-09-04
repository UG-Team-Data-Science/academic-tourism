# Academic Tourism Entity Annotation Project

Pipeline for creating and testing entity annotations on Dutch historical texts using LLMs.

## Workflow

### 1. Preprocessing (`preprocess.ipynb`)
- Extracts entities from DOCX files with **bold text** and **(label)** annotations
- Uses `wtpsplit` for Dutch sentence segmentation
- Outputs to `data/examples_to_clean.jsonl`

### 2. Label Cleanup (`cleanup_labels.ipynb`)
- Standardizes entity labels (e.g., "E 53 Place" → "E53 Place")
- Filters to ontology classes starting with E, F, or P
- Outputs to `data/examples_cleaned.jsonl`

### 3. Few-shot Prediction (`fewshot_test.ipynb`)
- Uses `multilingual-e5-base` embeddings for similarity matching
- Selects 5 similar training examples per test case
- Tests Mistral Small 3.2 and GPT-4o-mini models
- Outputs predictions to `data/llm_predictions.jsonl`

## Setup

Install dependencies with `uv`:
```bash
uv sync
```

## Usage

1. Place annotated DOCX files in `data/`
2. Run notebooks in order: `preprocess.ipynb` → `cleanup_labels.ipynb` → `fewshot_test.ipynb`
3. Generate HTML report: `python entity_markup_converter.py data/llm_predictions.jsonl report.html`

## Entity Types
- E53 Place, E21 Person, E19 Physical Thing, E52 Time-Span
- E54 Dimension, E86 Leaving, E74 Group, E9 Move

## Files
```
├── preprocess.ipynb              # Extract annotations
├── cleanup_labels.ipynb          # Clean labels  
├── fewshot_test.ipynb           # Generate predictions
├── entity_markup_converter.py   # Create HTML report
└── data/                        # Input DOCX + output JSONL files
```

The pipeline processes manually annotated Dutch texts and generates comparative LLM predictions using few-shot learning.
