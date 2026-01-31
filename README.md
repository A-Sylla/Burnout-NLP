

# Early Identification of Occupational Burnout through Text-Based Emotion Analysis

This repository contains the analysis pipeline and code used to investigate early indicators of occupational burnout from text data, leveraging natural language processing (NLP) and transformer-based models.

The study focuses on detecting emotional patterns in textual data (e.g., surveys, journals, messages) to support early intervention strategies for burnout in workplace settings.

---

## Project Overview

**Goal:** Automatically identify early signs of occupational burnout through emotion classification in text using state-of-the-art NLP models.

**Key Features:**

* Uses transformer-based models (BERT) for emotion classification.
* Supports multi-class emotion prediction (e.g., stress, sadness, frustration, etc.).
* Generates visualizations of emotion distributions and confusion matrices.
* Provides reproducible pipelines for model training, evaluation, and interpretation.

**Data Focus:** Occupational texts annotated with emotion labels.

---

## Scripts

### `train_model.py`

* Loads training, validation, and test datasets.
* Cleans and preprocesses text (removal of punctuation, lowercasing).
* Tokenizes text using BERT tokenizer.
* Maps emotion labels to integer classes.
* Trains BERT for sequence classification using Hugging Face `Trainer`.
* Evaluates model performance on a test set.
* Outputs:

  * Confusion matrix visualizations
  * Classification reports
  * Emotion distribution plots

---

## Methods

### Models Evaluated

* BERT-base-uncased for sequence classification.

### Training Strategies

* Early stopping and evaluation per epoch.
* Weight decay and warmup steps to improve optimization.
* Fixed random seed for reproducibility.

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score (per class)
* Confusion matrices

### Interpretability

* Visualizations of emotion distributions.
* Confusion matrices to identify misclassification patterns.

---

## Reproducibility and Data

* Raw datasets (`train.txt`, `val.txt`, `test.txt`) are expected in a text format with `text;emotion` per line.
* The repository provides a fully reproducible pipeline: users can train and evaluate models with their own data formatted according to the expected structure.
* All steps including preprocessing, tokenization, model training, evaluation, and visualization are included.

---

## Quickstart / Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages include:

* `pandas`, `numpy`, `scikit-learn`
* `torch`, `transformers`, `datasets`
* `matplotlib`, `seaborn`

---

### 2. Prepare your data

* Provide text datasets with `text;emotion` format.
* Ensure emotion labels in your validation/test datasets match those in training.

---

### 3. Train the model

```bash
python src/train_model.py
```

This script will:

* Load and clean text data
* Tokenize text using BERT tokenizer
* Map emotion labels to integers
* Train the BERT model
* Evaluate performance on the test set
* Save results in `results/` (plots, logs, confusion matrices)

---

### 4. Outputs

Key outputs saved in `results/` include:

* `Figure_2_Emotion_Distribution.tiff` – distribution of emotion classes
* Confusion matrix figures for test predictions
* Classification report printed in console
* Hugging Face logs in `results/logs/`

---

### 5. Optional Extensions

* Fine-tune BERT hyperparameters (learning rate, batch size, number of epochs)
* Apply additional preprocessing (emoji handling, stopwords removal)
* Save trained model and tokenizer for later inference:

```python
model.save_pretrained("./results/bert_emotion_model")
tokenizer.save_pretrained("./results/bert_emotion_model")
```
