  
**Sentiment Analysis on Twitter tweets** using classical Natural Language Processing (NLP) techniques and Machine Learning.    
This project cleans and preprocesses tweets, converts text to TF-IDF features, trains baseline classifiers (MultinomialNB & Logistic Regression), and provides an inference pipeline for new tweets.

\---

 **Project Summary**  
\- Dataset: **Sentiment140** (1.6M tweets; binary labels 0 \= negative, 4 \= positive). A sampled subset was used for experimentation.  
\- Main steps: Data cleaning → Tokenization → Stopword removal → Lemmatization → TF-IDF vectorization → Model training → Evaluation → Inference.  
\- Models: Multinomial Naive Bayes, Logistic Regression (best performer).  
\- Final deliverables: Jupyter Notebook (\`NLP\_PROJECT.ipynb\`), trained artifacts (TF-IDF vectorizer & model), and inference helper.

  **Features**  
\- Robust tweet preprocessing (URLs, mentions, hashtags removed; lemmatization; stopword removal).  
\- TF-IDF feature extraction with n-grams (unigrams \+ bigrams).  
\- Baseline ML models with evaluation (accuracy, precision, recall, F1, confusion matrix).  
\- Inference script/function for classifying new tweets.  
\- Guidance for upgrading to transformer-based models (DistilBERT/BERT).

\---

  **Repository Structure**

twitter-sentiment-analysis-nlp/  
 ├── data/  
 │ └── training.1600000.processed.noemoticon.csv \# (not stored in repo; download separately)  
 ├── notebooks/  
 │ └── NLP\_PROJECT.ipynb  
 ├── models/  
 │ ├── tfidf\_vectorizer.joblib  
 │ └── logistic\_model.joblib  
 ├── src/  
 │ └── predict\_helper.py \# inference helper (optional)  
 ├── README.md  
 └── requirements.txt

\---

  **Tech Stack**  
\- Python 3.x    
\- pandas, numpy    
\- scikit-learn    
\- nltk, spacy (optional)    
\- joblib    
\- matplotlib, seaborn

—

  **Installation**

1\. Clone the repo:  
\`\`\`bash  
git clone https://github.com/\<your-username\>/twitter-sentiment-analysis-nlp.git  
cd twitter-sentiment-analysis-nlp

2. Create a virtual environment (recommended) and install dependencies:

python \-m venv venv  
source venv/bin/activate        \# macOS/Linux  
\# venv\\Scripts\\activate         \# Windows

pip install \-r requirements.txt

**Example `requirements.txt`** (recommended):

pandas  
numpy  
scikit-learn  
nltk  
joblib  
matplotlib  
seaborn  
spacy

3. (NLTK / SpaCy setup) In a Python shell or notebook run:

import nltk  
nltk.download('punkt')  
nltk.download('stopwords')  
nltk.download('wordnet')  
\# If using spaCy:  
\# python \-m spacy download en\_core\_web\_sm

---

##  **Dataset**

The project uses the **Sentiment140** dataset (available publicly). The file name used in the notebook: [https://drive.google.com/file/d/1sSn01W6Mxv8vnJf0HTe-rWDWFBG0-9ng/view?usp=sharing](https://drive.google.com/file/d/1sSn01W6Mxv8vnJf0HTe-rWDWFBG0-9ng/view?usp=sharing) 

##  **How to Run (Notebook)**

1. Open `notebooks/NLP_PROJECT.ipynb` in Jupyter Notebook / JupyterLab / Colab.

2. Update `FILE_PATH` at the top of the notebook to point to your dataset location (e.g., `data/training.1600000.processed.noemoticon.csv`).

3. Run cells sequentially. Key sections:

   * Imports & setup

   * Load & sample dataset

   * Preprocessing (cleaning, tokenization, lemmatization)

   * Vectorization (TF-IDF)

   * Train/Test split & model training

   * Evaluation & inference

---

##  **Quick Usage Example (Inference)**

If you saved the vectorizer and model (`models/tfidf_vectorizer.joblib`, `models/logistic_model.joblib`), you can run:

import joblib  
from src.predict\_helper import predict\_sentiment\_texts\_safe  \# optional helper

tfidf \= joblib.load("models/tfidf\_vectorizer.joblib")  
lr \= joblib.load("models/logistic\_model.joblib")

texts \= \[  
    "I love this new phone\! Amazing battery life.",  
    "Terrible customer service — will never buy again."  
\]

preds, probs \= predict\_sentiment\_texts\_safe(texts, model=lr, vectorizer=tfidf)  
for t, p in zip(texts, preds):  
    print(t, "-\>", "Positive" if p==1 else "Negative")

---

## **📈 Results Summary**

* Logistic Regression accuracy: **\~0.89–0.90** (varies depending on sampled data and preprocessing).  
* MultinomialNB accuracy: **\~0.84–0.86**.  
* Key failure cases: sarcastic and neutral tweets, slang, and emoji-heavy tweets.

---

##  **Next Steps / Enhancements**

* Fine-tune a transformer (DistilBERT / BERT) for improved contextual understanding.  
* Add neutral class (multi-class classification).  
* Deploy as a web app with Streamlit or FastAPI.  
* Integrate realtime data collection from Twitter API for streaming analysis.

##  **License & Attribution**

* Dataset: Sentiment140 (please respect dataset terms).  
* Code: MIT License (or choose your preferred license).  
* Author: **Pawan Yadav** (or update to your name)

---

##  **Contact**

If you have questions contact:  
 **Pawan Yadav** — (mgsaipawanyadav@gmail.com)

