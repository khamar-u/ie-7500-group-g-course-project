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

# Fake News Detection: Data Preprocessing and Model Development

## Data Preprocessing  

### Objective  
This preprocessing pipeline is designed to **clean, transform, and prepare textual data** for machine learning models to detect fake news. The dataset is sourced from **Kaggle’s Fake News Detection Dataset**, which consists of two CSV files:  
- **True.csv** – Contains real news articles  
- **Fake.csv** – Contains fabricated news articles  

The preprocessing ensures that text data is structured and converted into numerical representations for effective model training.  

---

### Dataset Overview  
The dataset includes the following columns:  
- `title` – Headline of the article  
- `text` – Full content of the article  
- `subject` – Category of news (e.g., politics, world news)  
- `date` – Publication date  

To facilitate **binary classification**, an additional `label` column is added:  
- `1` → Real news  
- `0` → Fake news  

---

### Preprocessing Steps  

#### **1. Data Loading**  
- The dataset is loaded using `pandas.read_csv()`.  
- Labels are assigned, and both datasets are merged to form a **single structured dataset**.  
- The dataset is **shuffled** to prevent order bias and ensure a balanced training/testing split.  

#### **2. Text Cleaning**  
Raw text is often noisy and inconsistent. The following steps are applied:  
- **Lowercasing** – Converts all text to lowercase for uniformity.  
- **Removing URLs** – Deletes hyperlinks using regular expressions.  
- **Punctuation & Number Removal** – Eliminates unnecessary symbols.  
- **Stopword Removal** – Filters out common words (`nltk.stopwords`).  
- **Lemmatization** – Converts words to their base form using `WordNetLemmatizer` (e.g., *running* → *run*).  

#### **3. Feature Engineering with TF-IDF**  
After cleaning, textual data is vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert words into numerical features.  
A `TfidfVectorizer` is applied with `max_features=5000` to:  
- Retain the **top 5000 most informative words**  
- Reduce **dimensionality** while preserving important linguistic patterns  
- Weigh words based on their **importance across all documents**  

#### **4. Splitting into Training and Testing Sets**  
To evaluate model performance, the dataset is split into:  
- **80% Training Data** – For model training  
- **20% Testing Data** – For performance evaluation  

This ensures that the model generalizes well to unseen data.  

#### **5. Data Storage & Reusability**  
To facilitate reusability and avoid redundant processing, the transformed dataset is stored using Python’s `pickle` module:  
- `tfidf_vectorizer.pkl` – Stores the trained TF-IDF vectorizer  
- `train_test_data.pkl` – Stores the transformed train-test split  

By saving these components, future models can be trained without repeating preprocessing, **saving time and computational resources**.  

---

## Model Development  

To detect fake news, we implemented two different models:  
1. **Baseline Model** – Using **TF-IDF** and **Logistic Regression**  
2. **Advanced Model** – Using **BERT (Bidirectional Encoder Representations from Transformers)**  

### **Baseline Model: Logistic Regression + TF-IDF**  

#### **1. Feature Extraction**  
- TF-IDF was applied to extract the **5000 most relevant words** from the dataset.  
- The extracted words were converted into a **matrix representation**, allowing the model to weigh the importance of each word.  

#### **2. Model Training**  
- We trained the model using **Logistic Regression with GridSearchCV** to optimize hyperparameters.  
- **Cross-validation** helped in selecting the best parameters for higher accuracy.  

#### **3. Evaluation Metrics**  
The performance of the baseline model was evaluated using:  
- **Accuracy, Precision, Recall, F1-score**  
- **Confusion Matrix**  
- **ROC-AUC Curve**  

#### **4. Results**  


#### **Cross-Validation & Model Performance**  

The logistic regression model was evaluated using **5-fold cross-validation**, and the best parameters were selected through `GridSearchCV`. The results are as follows:  

- **5-fold cross-validation accuracy:** **99.45%**  
- **Test accuracy:** **99.56%**  
- **High precision (99.51%)** ensures fewer false positives.  
- **High recall (99.58%)** indicates minimal false negatives.  
- **Balanced F1-score (99.54%)** confirms strong performance.  

---

#### **Confusion Matrix & Model Reliability**  

The **Confusion Matrix** below shows that:  
- **Most fake news was correctly labeled as fake.**  
- **Most real news was accurately classified as real.**  

This confirms that **misclassifications were minimal**, ensuring high **model reliability**.

---

#### **ROC Curve Analysis**  

The **Receiver Operating Characteristic (ROC) curve** was used to evaluate the trade-off between the **true positive rate (TPR)** and the **false positive rate (FPR)**.  

- The **curve's shape** indicates:  
  ✅ **High true positive rate**  
  ✅ **Very low false positive rate**  

A well-defined **ROC curve close to (1,1)** suggests that the **model distinguishes fake vs. real news exceptionally well**.

---

### **Key Takeaways**
✔ The **baseline model achieved 99.56% accuracy**, demonstrating exceptional performance.  
✔ **Precision and recall balance ensures minimal false positives/negatives.**  
✔ **Confusion Matrix confirms correct classification of real and fake news.**  
✔ **The ROC curve validates that the model effectively differentiates classes.**  


#### Usage

Open the notebook and run the cells to execute the data loading, preprocessing, and model training steps:
```bash
jupyter notebook notebooks/NLP_Project.ipynb
