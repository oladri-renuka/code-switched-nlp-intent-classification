Code-Switched NLP Intent Classification
System for intent classification in code-switched (Hindi-English and Spanish-English) text using XLM-RoBERTa with supervised contrastive learning.

Overview
This project addresses the challenge of intent classification in code-switched multilingual text—a common phenomenon in social media and online communities where speakers naturally mix English with Hindi or Spanish within the same utterance. We fine-tune XLM-RoBERTa-large with a novel supervised contrastive loss mechanism to achieve state-of-the-art performance on this task.

Key Results
Model	Accuracy	Precision	Recall	F1 (Macro)
Contrastive (Fine-tuned)	91.62%	0.9170	0.9163	0.9167
Baseline (No Contrastive)	75.94%	0.7621	0.7638	0.7629
Improvement	+15.68%	+20.3%	+19.9%	+20.1%
Per-Class Performance (Contrastive Model)
Intent Class	Precision	Recall	F1-Score	Support
Complaint	0.95	0.93	0.94	351
Recommendation	0.90	0.90	0.90	344
Informational	0.91	0.91	0.91	427
Dataset
Source: SemEval 2020 Task 9 - Sentiment Analysis for Code-Mixed Social Media Text

Total Samples: 11,207 (after cleaning and deduplication)
Train/Val/Test Split: 80% / 10% / 10%
Language Pairs:
Hindi-English (primary)
Spanish-English (secondary)
Intent Classes: 3
Complaint (31.1%)
Recommendation (31.6%)
Informational (37.3%)
Data Processing Pipeline:

Loaded 17,594 SemEval samples
Verified code-switching using langdetect (12,367 valid samples, 70.2%)
Removed duplicates and short texts (<3 words)
Final cleaned dataset: 11,207 samples
Model Architecture
Base Model
Backbone: XLM-RoBERTa-large (568M parameters)
Tokenizer: SentencePiece (250K vocabulary)
Max Sequence Length: 256 tokens
Hidden Dimension: 1024
Attention Heads: 16
Encoder Layers: 24
Contrastive Fine-tuning
Embedding Projection: 1024 → 256 dimensions
Contrastive Loss Temperature: τ = 0.07
Loss Weighting: λ = 0.1 (contrastive weight)
Total Loss: L = CE_loss + λ × contrastive_loss
Optimizer: AdamW (lr=2e-5)
Scheduler: Linear warmup (500 steps)
Batch Size: 32
Training Epochs: 10 (with early stopping, patience=3)
Classification Head
Input: [CLS] token representation (1024-dim)
Dropout: 0.1
Output Layer: Dense (3 classes)
Activation: Softmax
Installation
Requirements
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
