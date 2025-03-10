# IE 7500 Group G Course Project: Fake News Detection

## Overview
This project aims to detect fake news using Natural Language Processing (NLP) techniques. It includes data preprocessing, feature extraction using TF-IDF, and model training using Logistic Regression and BERT fine-tuning.

## Directory Structure
ie-7500-group-g-course-project/ ├── README.md ├── .gitignore ├── src/ │ ├── data_loading_and_preprocessing.py │ ├── tfidf_baseline_model.py │ └── bert_fine_tuning.py ├── data/ │ ├── True.csv │ └── Fake.csv ├── models/ │ ├── best_logreg_model.pkl │ └── fine_tuned_bert/ │ ├── config.json │ ├── pytorch_model.bin │ └── tokenizer_config.json ├── docs/ │ └── documentation.md

## Data Loading and Preprocessing

The script `data_loading_and_preprocessing.py` loads and preprocesses the news datasets, including cleaning the text and combining the datasets.

## Model Training

### Baseline Model

The script `tfidf_baseline_model.py` trains a logistic regression model with hyperparameter tuning using TF-IDF features.

### Advanced Model

The script `bert_fine_tuning.py` fine-tunes a BERT model for classification.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
