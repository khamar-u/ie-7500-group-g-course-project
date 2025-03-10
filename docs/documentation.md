# Documentation

## Project Overview

This project aims to classify news articles as true or fake using machine learning models. It includes data preprocessing, a baseline logistic regression model, and an advanced BERT model for classification.

## Data Loading, Preprocessing, and Model Training

### Notebook: `NLP_Project.ipynb`

#### Overview

This Jupyter Notebook performs the following steps:
1. **Load Datasets**: Reads the true and fake news datasets from the zipped CSV files.
2. **Label Datasets**: Adds labels to the datasets (1 for true news, 0 for fake news).
3. **Combine Datasets**: Merges the true and fake news datasets into a single dataframe.
4. **Shuffle Data**: Randomly shuffles the combined dataset to ensure a random distribution.
5. **Clean Text**: Cleans the text data by:
   - Converting text to lowercase
   - Removing URLs, punctuation, and numbers
   - Removing stopwords and lemmatizing words
6. **Train Baseline Model**: Trains a logistic regression model with hyperparameter tuning using TF-IDF features.
7. **Fine-Tune BERT Model**: Fine-tunes a BERT model for classification.
8. **Evaluate Models**: Evaluates the models on the test set and calculates accuracy, precision, recall, and F1 score.
9. **Save Models**: Saves the trained models to disk.
10. **Generate Plots**: Generates and saves the confusion matrix and ROC curve.

#### Usage

Open the notebook and run the cells to execute the data loading, preprocessing, and model training steps:
```bash
jupyter notebook notebooks/NLP_Project.ipynb
