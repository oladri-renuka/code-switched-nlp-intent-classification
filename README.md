# Code-Switched NLP: Intent Classification for Multilingual Social Media

A production-ready system for classifying intents in code-switched text (Hindi-English and Spanish-English) using XLM-RoBERTa with contrastive learning.

## Project Overview

This project addresses the challenge of intent classification in code-switched text (text that mixes multiple languages mid-sentence). We collect real Reddit data, annotate it with Label Studio, and fine-tune XLM-RoBERTa-large with supervised contrastive loss to outperform standard baselines.

### Key Components

1. **Data Collection**: Reddit posts/comments (r/hinglish, r/latinos, r/IndiansInUSA)
2. **Annotation Pipeline**: Label Studio + LLM-assisted annotation (Claude Sonnet)
3. **Model Training**: XLM-RoBERTa-large with contrastive loss
4. **Evaluation**: Comparison against baselines (base model, Google Translate)
5. **Deployment**: FastAPI + Gradio demo

## Directory Structure

```
Code_Switched_NLP/
├── data/
│   ├── raw/                    # Raw Reddit data (JSON)
│   ├── annotated/              # Label Studio export
│   └── processed/              # Train/val/test splits
├── scripts/
│   ├── 01_collect_reddit_data.py
│   ├── 02_verify_codeswitching.py
│   ├── 03_prepare_label_studio.py
│   ├── 04_llm_annotation.py
│   ├── 05_agreement_analysis.py
│   └── 06_train_model.py
├── models/
│   ├── base_model.py           # Model architecture with contrastive loss
│   ├── trainer.py              # Custom Hugging Face Trainer
│   └── checkpoints/            # Saved model checkpoints
├── evaluation/
│   ├── baseline_comparison.py
│   └── results/
├── deployment/
│   ├── api.py                  # FastAPI server
│   ├── gradio_demo.py
│   └── requirements.txt
├── notebooks/
│   └── exploration.ipynb
├── config.yaml                 # Project configuration
└── README.md
```

## Technology Stack

- **Base Model**: `xlm-roberta-large` (HuggingFace)
- **Training Framework**: Transformers + PyTorch
- **Data Annotation**: Label Studio (self-hosted)
- **Language Detection**: langdetect
- **Data Source**: PRAW (Reddit API)
- **Metrics**: seqeval, scikit-learn
- **Deployment**: FastAPI + Gradio
- **LLM Annotation**: Claude Sonnet 4.6

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Reddit Data
```bash
python scripts/01_collect_reddit_data.py
```

### 3. Launch Label Studio
```bash
docker run -it -p 8080:8080 heartexlabs/label-studio
```

### 4. Train Model
```bash
python scripts/06_train_model.py
```

### 5. Deploy
```bash
# Start FastAPI
python deployment/api.py

# In another terminal, start Gradio demo
python deployment/gradio_demo.py
```

## Datasets & Baselines

**Data Requirements**:
- Minimum 3,000 examples with code-switching (Hindi-English or Spanish-English)
- Manual annotation: 500+ examples
- LLM-assisted annotation: 2,500+ examples with Cohen's kappa agreement

**Baselines**:
1. XLM-RoBERTa-large (base, no fine-tuning)
2. Google Translate → English classification
3. XLM-RoBERTa-large + Contrastive Loss (this work)

## Expected Results

- Improvement in macro F1 over baselines on code-switched intent classification
- Analysis of code-switching patterns and their effect on intent classification
- Deployment demo comparing base vs. fine-tuned model

## References

- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- [Code-Switching in NLP](https://aclanthology.org/2021.acl-long.148/)

---

**Status**: Project initialization
**Last Updated**: 2026-09-13
