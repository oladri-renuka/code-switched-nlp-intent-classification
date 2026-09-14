# Code-Switched NLP — Intent Classification
 
**XLM-RoBERTa with supervised contrastive loss achieves 91.62% F1 on code-switched Hindi-English and Spanish-English text — a 15.68-point improvement over the fine-tuned baseline.**
 
---
 
## Results
 
| Model | Accuracy | F1 (Weighted) | Improvement |
|-------|----------|---------------|-------------|
| XLM-RoBERTa + Contrastive Loss | **91.62%** | **0.9167** | — |
| XLM-RoBERTa Baseline | 75.94% | 0.7629 | +15.68% |
 
**Per-class breakdown (contrastive model):**
 
| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Complaint | 0.95 | 0.93 | 0.94 | 351 |
| Recommendation | 0.90 | 0.90 | 0.90 | 344 |
| Informational | 0.91 | 0.91 | 0.91 | 427 |
 
---
 
## What This Is
 
Most NLP systems assume monolingual input. Real social media data does not — users naturally mix languages mid-sentence ("Bhai, tumhe pata hai ki naya iPhone release ho gaya?"). Standard intent classifiers fail on this because they were not trained on code-switched patterns.
 
This project fine-tunes XLM-RoBERTa-large with supervised contrastive loss to learn language-agnostic intent representations — the same intent expressed in Hindi-English or Spanish-English should have similar embeddings regardless of which language is used for which words.
 
---
 
## Why Contrastive Loss Works Here
 
Standard cross-entropy fine-tuning treats each example independently. Contrastive loss additionally pulls together examples of the same intent class across language mixes and pushes apart examples of different intents.
 
```
Total Loss = CE_loss + 0.1 × contrastive_loss
Temperature: τ = 0.07
Projection: 1024 → 256 dimensions
```
 
The result: the model learns that "This product is terrible, bilkul bekar hai" (complaint, Hindi-English) and "Este producto es muy malo, totally waste of money" (complaint, Spanish-English) should have similar representations, even though they share few tokens.
 
---
 
## Dataset
 
**Source:** SemEval-2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text
 
| Property | Value |
|----------|-------|
| Total samples | 11,207 (after cleaning) |
| Language pairs | Hindi-English, Spanish-English |
| Split | 80 / 10 / 10 (train / val / test) |
| Code-switching verified | Yes — langdetect filter (70.2% pass rate) |
 
**Class distribution (balanced):**
- Informational: 37.3%
- Recommendation: 31.6%
- Complaint: 31.1%
**Data pipeline:** 17,594 raw SemEval samples → langdetect verification → duplicate removal → short text filter → 11,207 clean examples.
 
---
 
## Model Architecture
 
```
Input text (code-switched)
       │
XLM-RoBERTa-large (568M params)
  24 layers, 16 attention heads
  SentencePiece tokenizer (250K vocab)
  Max sequence length: 256 tokens
       │
    [CLS] token (1024-dim)
       │
  ┌────┴────┐
  │         │
Classification    Contrastive projection
head (3 classes)  1024 → 256 dims
  │         │
  CE loss   Contrastive loss (τ=0.07)
  └────┬────┘
    Total loss
```
 
**Training:** AdamW, lr=2e-5, linear warmup (500 steps), batch size 32, 10 epochs with early stopping (patience=3). Trained on A100 in ~24 minutes.
 
---
 
## Setup
 
```bash
git clone https://github.com/oladri-renuka/code-switched-nlp-intent-classification
cd code-switched-nlp-intent-classification
pip install -r requirements.txt
```
 
**Quick prediction:**
```bash
python predict.py
```
 
```
Text: Bhai, tumhe pata hai ki naya iPhone release ho gaya?
  Contrastive: recommendation (87.3%)
  Baseline:    recommendation (72.1%)
```
 
**API:**
```bash
python deployment/api.py
# Docs at http://localhost:8000/docs
```
 
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Bhai, ye product bilkul bekar hai, total waste of money"}'
```
 
```json
{
  "contrastive_prediction": "complaint",
  "contrastive_confidence": 0.934,
  "baseline_prediction": "complaint",
  "baseline_confidence": 0.731,
  "languages": ["en", "hi"]
}
```
 
**Streamlit interface:**
```bash
streamlit run deployment/streamlit_app.py
```
 
---
 
## Training From Scratch
 
```bash
python scripts/prepare_data.py SemEval2020-Task9
python scripts/train.py
python scripts/evaluate.py
```
 
Edit `config.yaml` to adjust hyperparameters:
 
```yaml
model:
  base_model: "xlm-roberta-large"
  contrastive_loss:
    enabled: true
    temperature: 0.07
    weight: 0.1
  training:
    epochs: 10
    batch_size: 32
    learning_rate: 2e-5
```
 
---
 
## File Structure
 
```
code-switched-nlp-intent-classification/
├── scripts/
│   ├── prepare_data.py       # Data loading, verification, cleaning
│   ├── train.py              # Train contrastive + baseline models
│   ├── evaluate.py           # Test set evaluation with per-class metrics
│   └── predict.py            # Single-text prediction
├── models/
│   ├── base_model.py         # XLM-RoBERTa + contrastive loss head
│   └── checkpoints/
├── deployment/
│   ├── api.py                # FastAPI REST endpoint
│   └── streamlit_app.py      # Web interface
├── data/
│   ├── raw/                  # SemEval source files
│   └── processed/            # Cleaned dataset
├── config.yaml
├── requirements.txt
└── README.md
```
 
---
 
## Tech Stack
 
| Component | Technology |
|-----------|------------|
| Base model | XLM-RoBERTa-large (HuggingFace) |
| Training | PyTorch, AdamW |
| Data verification | langdetect |
| API | FastAPI |
| Interface | Streamlit |
| Dataset | SemEval-2020 Task 9 |
 
---
 
## Citation
 
```bibtex
@inproceedings{joshi2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Joshi, Pratik and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation},
  year={2020}
}
```
  
---
 
## Dataset
 
**Source:** SemEval-2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text
 
| Property | Value |
|----------|-------|
| Total samples | 11,207 (after cleaning) |
| Language pairs | Hindi-English, Spanish-English |
| Split | 80 / 10 / 10 (train / val / test) |
| Code-switching verified | Yes — langdetect filter (70.2% pass rate) |
 
**Class distribution (balanced):**
- Informational: 37.3%
- Recommendation: 31.6%
- Complaint: 31.1%
**Data pipeline:** 17,594 raw SemEval samples → langdetect verification → duplicate removal → short text filter → 11,207 clean examples.
 
---
 
## Model Architecture
 
```
Input text (code-switched)
       │
XLM-RoBERTa-large (568M params)
  24 layers, 16 attention heads
  SentencePiece tokenizer (250K vocab)
  Max sequence length: 256 tokens
       │
    [CLS] token (1024-dim)
       │
  ┌────┴────┐
  │         │
Classification    Contrastive projection
head (3 classes)  1024 → 256 dims
  │         │
  CE loss   Contrastive loss (τ=0.07)
  └────┬────┘
    Total loss
```
 
**Training:** AdamW, lr=2e-5, linear warmup (500 steps), batch size 32, 10 epochs with early stopping (patience=3). Trained on A100 in ~24 minutes.
 
---
 
## Setup
 
```bash
git clone https://github.com/oladri-renuka/code-switched-nlp-intent-classification
cd code-switched-nlp-intent-classification
pip install -r requirements.txt
```
 
**Quick prediction:**
```bash
python predict.py
```
 
```
Text: Bhai, tumhe pata hai ki naya iPhone release ho gaya?
  Contrastive: recommendation (87.3%)
  Baseline:    recommendation (72.1%)
```
 
**API:**
```bash
python deployment/api.py
# Docs at http://localhost:8000/docs
```
 
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Bhai, ye product bilkul bekar hai, total waste of money"}'
```
 
```json
{
  "contrastive_prediction": "complaint",
  "contrastive_confidence": 0.934,
  "baseline_prediction": "complaint",
  "baseline_confidence": 0.731,
  "languages": ["en", "hi"]
}
```
 
**Streamlit interface:**
```bash
streamlit run deployment/streamlit_app.py
```
 
---
 
## Training From Scratch
 
```bash
python scripts/prepare_data.py SemEval2020-Task9
python scripts/train.py
python scripts/evaluate.py
```
 
Edit `config.yaml` to adjust hyperparameters:
 
```yaml
model:
  base_model: "xlm-roberta-large"
  contrastive_loss:
    enabled: true
    temperature: 0.07
    weight: 0.1
  training:
    epochs: 10
    batch_size: 32
    learning_rate: 2e-5
```
 
---
 
## File Structure
 
```
code-switched-nlp-intent-classification/
├── scripts/
│   ├── prepare_data.py       # Data loading, verification, cleaning
│   ├── train.py              # Train contrastive + baseline models
│   ├── evaluate.py           # Test set evaluation with per-class metrics
│   └── predict.py            # Single-text prediction
├── models/
│   ├── base_model.py         # XLM-RoBERTa + contrastive loss head
│   └── checkpoints/
├── deployment/
│   ├── api.py                # FastAPI REST endpoint
│   └── streamlit_app.py      # Web interface
├── data/
│   ├── raw/                  # SemEval source files
│   └── processed/            # Cleaned dataset
├── config.yaml
├── requirements.txt
└── README.md
```
 
---
 
## Tech Stack
 
| Component | Technology |
|-----------|------------|
| Base model | XLM-RoBERTa-large (HuggingFace) |
| Training | PyTorch, AdamW |
| Data verification | langdetect |
| API | FastAPI |
| Interface | Streamlit |
| Dataset | SemEval-2020 Task 9 |
 
---
 
## Citation
 
```bibtex
@inproceedings{joshi2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Joshi, Pratik and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation},
  year={2020}
}
```
 
---

## Dataset

**Source:** SemEval-2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text

| Property | Value |
|----------|-------|
| Total samples | 11,207 (after cleaning) |
| Language pairs | Hindi-English, Spanish-English |
| Split | 80 / 10 / 10 (train / val / test) |
| Code-switching verified | Yes — langdetect filter (70.2% pass rate) |

**Class distribution (balanced):**
- Informational: 37.3%
- Recommendation: 31.6%
- Complaint: 31.1%

**Data pipeline:** 17,594 raw SemEval samples → langdetect verification → duplicate removal → short text filter → 11,207 clean examples.

---

## Model Architecture

```
Input text (code-switched)
       │
XLM-RoBERTa-large (568M params)
  24 layers, 16 attention heads
  SentencePiece tokenizer (250K vocab)
  Max sequence length: 256 tokens
       │
    [CLS] token (1024-dim)
       │
  ┌────┴────┐
  │         │
Classification    Contrastive projection
head (3 classes)  1024 → 256 dims
  │         │
  CE loss   Contrastive loss (τ=0.07)
  └────┬────┘
    Total loss
```

**Training:** AdamW, lr=2e-5, linear warmup (500 steps), batch size 32, 10 epochs with early stopping (patience=3). Trained on A100 in ~24 minutes.

---

## Setup

```bash
git clone https://github.com/oladri-renuka/code-switched-nlp-intent-classification
cd code-switched-nlp-intent-classification
pip install -r requirements.txt
```

**Quick prediction:**
```bash
python predict.py
```

```
Text: Bhai, tumhe pata hai ki naya iPhone release ho gaya?
  Contrastive: recommendation (87.3%)
  Baseline:    recommendation (72.1%)
```

**API:**
```bash
python deployment/api.py
# Docs at http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Bhai, ye product bilkul bekar hai, total waste of money"}'
```

```json
{
  "contrastive_prediction": "complaint",
  "contrastive_confidence": 0.934,
  "baseline_prediction": "complaint",
  "baseline_confidence": 0.731,
  "languages": ["en", "hi"]
}
```

**Streamlit interface:**
```bash
streamlit run deployment/streamlit_app.py
```

---

## Training From Scratch

```bash
python scripts/prepare_data.py SemEval2020-Task9
python scripts/train.py
python scripts/evaluate.py
```

Edit `config.yaml` to adjust hyperparameters:

```yaml
model:
  base_model: "xlm-roberta-large"
  contrastive_loss:
    enabled: true
    temperature: 0.07
    weight: 0.1
  training:
    epochs: 10
    batch_size: 32
    learning_rate: 2e-5
```

---

## File Structure

```
code-switched-nlp-intent-classification/
├── scripts/
│   ├── prepare_data.py       # Data loading, verification, cleaning
│   ├── train.py              # Train contrastive + baseline models
│   ├── evaluate.py           # Test set evaluation with per-class metrics
│   └── predict.py            # Single-text prediction
├── models/
│   ├── base_model.py         # XLM-RoBERTa + contrastive loss head
│   └── checkpoints/
├── deployment/
│   ├── api.py                # FastAPI REST endpoint
│   └── streamlit_app.py      # Web interface
├── data/
│   ├── raw/                  # SemEval source files
│   └── processed/            # Cleaned dataset
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | XLM-RoBERTa-large (HuggingFace) |
| Training | PyTorch, AdamW |
| Data verification | langdetect |
| API | FastAPI |
| Interface | Streamlit |
| Dataset | SemEval-2020 Task 9 |

---

## Citation

```bibtex
@inproceedings{joshi2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Joshi, Pratik and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation},
  year={2020}
}
```Requirements
Python 3.12+
CUDA 12.0+ (GPU recommended; CPU supported)
8GB+ RAM
Setup
# Clone repository
git clone https://github.com/oladri-renuka/code-switched-nlp-intent-classification.git
cd code-switched-nlp-intent-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Usage
Quick Prediction
python predict.py
Output:

Loading models...
✓ Ready!
Text: Bhai, tumhe pata hai ki naya iPhone release ho gaya?
  Contrastive: recommendation (87.3%)
  Baseline:    recommendation (72.1%)
Python API
from predict import load_models, predict

tokenizer, contrastive_model, baseline_model, device = load_models()

text = "Bhai, tumhe pata hai ki naya iPhone release ho gaya?"
pred, probs = predict(text, contrastive_model, tokenizer, device)

print(f"Prediction: {pred}")
print(f"Confidence: {probs.max():.2%}")
FastAPI Server
python deployment/api.py
# Server running on http://localhost:8000
# Docs: http://localhost:8000/docs
Endpoint: POST /classify

{
  "text": "Bhai, tumhe pata hai ki naya iPhone release ho gaya?",
  "compare_models": true
}
Response:

{
  "text": "Bhai, tumhe pata hai ki naya iPhone release ho gaya?",
  "languages": ["en", "hi"],
  "contrastive_prediction": "recommendation",
  "contrastive_confidence": 0.873,
  "baseline_prediction": "recommendation",
  "baseline_confidence": 0.721
}
Streamlit Interface
pip install streamlit
streamlit run deployment/streamlit_app.py
# Open http://localhost:8501
Evaluation
Test Set Performance
Command:

python scripts/evaluate.py
Results Summary:

Contrastive Model: 91.62% accuracy on 1,122 test samples
Baseline Model: 75.94% accuracy on same test set
Model Agreement: 78.1% (both models predict same class)
Metrics Computed
Accuracy (micro-averaged)
Precision, Recall, F1-Score (macro-averaged)
Per-class breakdown with support counts
Confusion matrices available in evaluation script
Training
To retrain the models from scratch:

# Prepare data
python scripts/prepare_data.py SemEval2020-Task9

# Train both models (contrastive + baseline)
python scripts/train.py

# Or train baseline only (skip contrastive)
python scripts/train_baseline_only.py
Training Details:

Hardware: A100 GPU (80GB VRAM)
Contrastive Training Time: ~24 minutes (10 epochs)
Baseline Training Time: ~16 minutes (6 epochs)
Checkpoints saved every epoch in models/checkpoints/
Configuration
Edit config.yaml to customize:

model:
  base_model: "xlm-roberta-large"
  contrastive_loss:
    enabled: true
    temperature: 0.07
    weight: 0.1
  training:
    epochs: 10
    batch_size: 32
    learning_rate: 2e-5
Project Structure
code-switched-nlp-intent-classification/
├── scripts/
│   ├── prepare_data.py          # Data loading, verification, cleaning
│   ├── train.py                 # Training both models
│   ├── train_baseline_only.py   # Baseline-only training
│   ├── evaluate.py              # Test set evaluation
│   └── predict.py               # Simple prediction script
├── models/
│   ├── base_model.py            # XLM-RoBERTa + contrastive loss
│   └── checkpoints/             # Trained model checkpoints
├── deployment/
│   ├── api.py                   # FastAPI server
│   └── streamlit_app.py         # Streamlit UI
├── data/
│   ├── raw/                     # Downloaded SemEval files
│   └── processed/               # Cleaned & prepared data
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
Citation
If you use this project, please cite:

@inproceedings{joshi2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Joshi, Pratik and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation},
  year={2020}
}
License
MIT License - See LICENSE file for details
