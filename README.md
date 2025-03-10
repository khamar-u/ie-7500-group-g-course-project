# IE 7500 Group G Course Project: Fake News Detection

## Overview
This project aims to detect fake news using Natural Language Processing (NLP) techniques. It includes data preprocessing, feature extraction using TF-IDF, and model training using Logistic Regression and BERT fine-tuning.

## Directory Structure
ie-7500-group-g-course-project/ ├── README.md ├── .gitignore ├── src/ │ ├── data_loading_and_preprocessing.py │ ├── tfidf_baseline_model.py │ └── bert_fine_tuning.py ├── data/ │ ├── True.csv │ └── Fake.csv ├── models/ │ ├── best_logreg_model.pkl │ └── fine_tuned_bert/ │ ├── config.json │ ├── pytorch_model.bin │ └── tokenizer_config.json ├── docs/ │ └── documentation.md

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/khamar-u/ie-7500-group-g-course-project.git
   cd ie-7500-group-g-course-project
2. Install the required packages:
   pip install -r requirements.txt
