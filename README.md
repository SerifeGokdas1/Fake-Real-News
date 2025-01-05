# Fake News Detection Project

This project focuses on detecting fake news using machine learning techniques. It involves preprocessing a dataset, tokenization, vectorization, and training multiple machine learning models to classify news as either true or fake.

---

## Table of Contents

- [About the Dataset](#about-the-dataset)
- [Project Workflow](#project-workflow)
- [Setup and Installation](#setup-and-installation)
- [Models and Evaluation](#models-and-evaluation)
- [Visualization Examples](#visualization-examples)
- [License](#license)

---

## About the Dataset

The dataset consists of **12,723 entries**, with an equal split between true and fake news. 
- **Number of True News Articles:** 6,194  
- **Number of Fake News Articles:** 6,529
Key attributes include:
- `Title`: The title of the news article.
- `Label`: The classification of the article (0 for fake, 1 for true).

### Fake News Sources:
- [Teyit.org](https://teyit.org/)
- [Doğruluk Payı](https://www.dogrulukpayi.com/dogruluk-kontrolleri)

### True News Sources:
- [BBC Türkçe](https://www.bbc.com/turkce)
- [Aposto](https://aposto.com/n/daily?tab=story)
- [Bianet](https://bianet.org/haberler)
- [Anadolu Ajansı (AA)](https://www.aa.com.tr/tr/gundem)
- [Deutsche Welle (DW) Türkiye](https://www.dw.com/tr/t%C3%BCrkiye)
- [Fatih Altaylı Köşe Yazıları](https://fatihaltayli.com.tr/kategori/kose-yazisi)

---

## Project Workflow

1. **Data Preprocessing**:
    - Cleaning text by removing punctuation, converting to lowercase, and eliminating stopwords.
    - Lemmatization of words to standardize forms.

2. **Tokenization**:
    - Splitting text into tokens using the `alibayram/tr_tokenizer`.

3. **Vectorization**:
    - Converting text data into numerical embeddings using the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model.

4. **Model Training**:
    - Models trained: Logistic Regression, Neural Networks, Random Forest, Support Vector Machine, and XGBoost.
    - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

---

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.10.0
- Required libraries: `pandas`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `transformers`, `sentence-transformers`

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-repo/fake-news-detection.git
cd fake-news-detection
```


Install the required dependencies:

`pip install -r requirements.txt`

Run the project locally:
`python main.py`

---

## Models and Evaluation

| Model                 | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | Macro Avg | Weighted Avg | ROC-AUC |
|------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|-----------|--------------|---------|
| Logistic Regression    | 0.88     | 0.90          | 0.85          | 0.85       | 0.90       | 0.88         | 0.88         | 0.88      | 0.88         | 0.92    |
| Neural Networks        | 0.89     | 0.89          | 0.89          | 0.89       | 0.88       | 0.89         | 0.88         | 0.89      | 0.89         | 0.91    |
| Random Forest          | 0.86     | 0.86          | 0.87          | 0.88       | 0.84       | 0.87         | 0.86         | 0.86      | 0.86         | 0.90    |
| Support Vector Machine | 0.87     | 0.90          | 0.85          | 0.84       | 0.90       | 0.87         | 0.87         | 0.87      | 0.87         | 0.91    |
| XGBoost                | 0.87     | 0.89          | 0.86          | 0.86       | 0.89       | 0.88         | 0.87         | 0.87      | 0.87         | 0.91    |


### Notes:
- **Precision (0)** and **Precision (1)** indicate how effectively the model identifies each class (fake or true).
- **Recall (0)** and **Recall (1)** show how many of the actual samples for each class are correctly identified.
- **F1-Score (0)** and **F1-Score (1)** represent the harmonic mean of precision and recall for each class.
- **Macro Avg** is the unweighted average of metrics across both classes.
- **Weighted Avg** takes into account the support (number of true instances) of each class.
- **ROC-AUC** measures the ability of the model to distinguish between classes; higher values indicate better performance.


---

## Visualization Examples

### Confusion Matrices:
Below are the confusion matrices for each model used:

![Logistic Regression Confusion Matrix](https://github.com/user-attachments/assets/54cf423b-3611-46f7-a690-6e29ea658ac5)

![Neural Networks Confusion Matrix](https://github.com/user-attachments/assets/682c580a-7dca-4736-9f32-613abc37d6ff)

![Random Forest Confusion Matrix](https://github.com/user-attachments/assets/88fd7b52-30a7-4ffa-9671-2353a5987727)

![SVM Confusion Matrix](https://github.com/user-attachments/assets/22b3d910-25cb-486c-8f7d-09d8fadbab08)

![XGBoost Confusion Matrix](https://github.com/user-attachments/assets/133088f8-5271-4f32-b42a-028c4e3dd781)

---

### ROC Curves:
The ROC curves for all models are shown below to compare their performance:

![Logistic Regression ROC Curve](https://github.com/user-attachments/assets/89ee663a-b574-4936-a1f7-97a71e525276)

![Neural Networks ROC Curve](https://github.com/user-attachments/assets/bd6b6e4a-2081-416a-b861-498c31b28f28)

![Random Forest ROC Curve](https://github.com/user-attachments/assets/5657de67-e61b-4709-96c3-ce67e472bc4e)

![SVM ROC Curve](https://github.com/user-attachments/assets/1ecefd09-1ced-417e-a46a-d0f8e37419f8)

![XGBoost ROC Curve](https://github.com/user-attachments/assets/4fc486d8-749a-4d66-bccf-31fc4792781b)


---

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.


