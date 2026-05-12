# News Classification using NLP

## 📌 Overview

This project classifies news articles into different categories using Natural Language Processing and Machine Learning.

The main goal of this project is to understand how text data can be cleaned, transformed into numerical features, and used for classification tasks.

---

## 🎯 Problem Statement

News articles are published in large numbers every day across different categories such as politics, sports, business, technology, and entertainment.

Manually categorizing news articles can be time-consuming. This project uses NLP and machine learning techniques to automatically classify news text into the correct category.

---

## 🧠 NLP Workflow

The project follows these steps:

1. Data collection
2. Text cleaning
3. Text preprocessing
4. TF-IDF vectorization
5. Model training using Multinomial Naive Bayes
6. Model evaluation
7. Prediction on new text

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Jupyter Notebook
- Flask

---

## 📂 Project Structure

    News_Classification/
    │
    ├── News_Classification_.ipynb
    ├── app.py
    ├── requirements.txt
    └── README.md

---

## 📊 Dataset

The dataset contains news articles along with their respective categories.

Each record generally includes:

- News text
- News category/label

The target variable represents the category of the news article.

---

## 🔤 Text Preprocessing

Text preprocessing is an important step in NLP. The following preprocessing techniques are used or can be applied:

- Converting text to lowercase
- Removing punctuation
- Removing stopwords
- Tokenization
- Removing unnecessary spaces
- Converting text into numerical form using TF-IDF

---

## 🔢 Feature Extraction

This project uses **TF-IDF Vectorization** to convert text data into numerical features.

TF-IDF helps represent the importance of words in a document while reducing the impact of commonly occurring words.

---

## 🤖 Model Used

This project uses **Multinomial Naive Bayes**, a commonly used machine learning algorithm for text classification tasks.

Multinomial Naive Bayes works well with text data because it is suitable for features like word counts and TF-IDF values.

In this project, the text data is converted into numerical features using **TF-IDF Vectorization**, and then the Multinomial Naive Bayes model is trained to classify news articles into different categories.

---

## 📈 Model Evaluation

The model achieved an accuracy of **90.6%** on the test data.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.91 | 0.90 | 0.91 | 7472 |
| 1 | 0.94 | 0.98 | 0.96 | 7560 |
| 2 | 0.88 | 0.88 | 0.88 | 7440 |
| 3 | 0.89 | 0.87 | 0.88 | 7528 |

### Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.906 |
| Macro Average Precision | 0.91 |
| Macro Average Recall | 0.91 |
| Macro Average F1-score | 0.91 |
| Weighted Average Precision | 0.91 |
| Weighted Average Recall | 0.91 |
| Weighted Average F1-score | 0.91 |

The model performs well overall, with the best performance on **Class 1**, which achieved an F1-score of **0.96**.

---

## 🚀 How to Run the Project

### 1. Clone the repository

    git clone https://github.com/SayyadRizwan/News_Classification.git

### 2. Move into the project folder

    cd News_Classification

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Run the application

    python app.py

---

## 🧪 Sample Prediction

Example input:

    The government announced new policies for economic growth.

Example output:

    Predicted Category: 0

---

## 📌 Key Learnings

Through this project, I learned:

- How to work with text data
- How to clean and preprocess text
- How to convert text into numerical features
- How TF-IDF works in NLP
- How Multinomial Naive Bayes is used for text classification
- How to evaluate NLP classification models using accuracy, precision, recall, and F1-score

---

## 🔮 Future Improvements

- Improve text preprocessing techniques
- Add more model comparisons
- Map numeric classes to actual category names
- Deploy the application online
- Add a better user interface
- Try transformer-based models like BERT

---

## 👨‍💻 Author

**Rizwan Sayyad**

GitHub: [SayyadRizwan](https://github.com/SayyadRizwan)
