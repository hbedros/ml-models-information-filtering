# Evaluating ML Models for Information Filtering and Truth Validation

This project evaluates machine learning models, specifically a hybrid approach combining Latent Dirichlet Allocation (LDA) and Bidirectional Encoder Representations from Transformers (BERT), to efficiently filter qualitative data and validate information accuracy.

## Overview

- **Goal:** Optimize qualitative data extraction, reducing irrelevant information before feeding data into Large Language Models (LLMs).
- **Approach:** Hybrid LDA + BERT model to capture thematic structures and semantic context.
- **Use Cases:** Filtering news articles, social media, internal documents.

## Repo Structure

- `data/` — Contains datasets for training and testing.
- `src/` — Python scripts for data processing, model training, and evaluation.
- `notebooks/` — Interactive notebooks for exploratory analysis and model evaluation.

## Quickstart

**Install dependencies:**
```bash
pip install -r requirements.txt
