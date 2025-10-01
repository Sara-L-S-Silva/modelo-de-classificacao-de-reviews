import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk

import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)

# baixa as stopwords para o pré-processamento
nltk.download("stopwords")
stopwords_pt = set(stopwords.words("portuguese"))

# --------------------------
# 1. Carrega e cria o rótulo (y)
# --------------------------
#df = pd.read_csv("olist_order_reviews_dataset.csv")
# 2. CONSTRUA O CAMINHO COMPLETO PARA O ARQUIVO CSV
csv_file_path = os.path.join(path, "olist_order_reviews_dataset.csv")

# 3. USE ESSE CAMINHO COMPLETO PARA LER O ARQUIVO
df = pd.read_csv(csv_file_path)

def score_to_label(score):
    if score >= 4:
        return "positivo"
    elif score <= 2:
        return "negativo"
    else:
        return None

df["label"] = df["review_score"].apply(score_to_label)
df = df.dropna(subset=["label"])  # descarta neutros

# --------------------------
# 2. Concatenar título + mensagem
# --------------------------
df["review_text"] = (df["review_comment_title"].fillna("") + " " +
                     df["review_comment_message"].fillna(""))

# --------------------------
# 3. Pré-processamento do texto
# --------------------------
def clean_text(text):
    text = text.lower()  # minúsculas
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # remove pontuação
    text = re.sub(r"\d+", " ", text)  # remove números
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_pt]  # remove stopwords
    return " ".join(tokens)

df["clean_text"] = df["review_text"].apply(clean_text)

# --------------------------
# 4. Separar treino/teste
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# --------------------------
# 5. Vetorização com TF-IDF
# --------------------------
# Ajustes: limitar features, aplicar min_df (ignorar termos muito raros)
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,5), min_df=5)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --------------------------
# 6. Modelos
# --------------------------

# Regressão Logística com regularização forte (C baixo)
logreg = LogisticRegression(max_iter=1000, C=0.5, penalty="l2", class_weight="balanced")
logreg.fit(X_train_tfidf, y_train)

y_pred_logreg = logreg.predict(X_test_tfidf)

def show_top_features(model, vectorizer, n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    if isinstance(model, LogisticRegression):
        coefs = model.coef_[0]
        top_pos = feature_names[np.argsort(coefs)[-n:]]
        top_neg = feature_names[np.argsort(coefs)[:n]]
    elif isinstance(model, MultinomialNB):
        log_prob = model.feature_log_prob_
        top_pos = feature_names[np.argsort(log_prob[1])[-n:]]
        top_neg = feature_names[np.argsort(log_prob[0])[-n:]]
    else:
        return
    print("🔹 Palavras indicativas de reviews positivas:", top_pos)
    print("🔸 Palavras indicativas de reviews negativas:", top_neg)
    
# Dados relevantes para a análise de eficácia do algoritmo de  Regressão Logística

print("\n=== Regressão Logística (ajustada contra overfitting) ===")
print(classification_report(y_test, y_pred_logreg))

# 7. Razões estruturadas (palavras mais relevantes)
print("\n=== Palavras mais relevantes (LogReg) ===")
show_top_features(logreg, tfidf)


# Naive Bayes 

nb = MultinomialNB(fit_prior=True)

nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# Dados relevantes para a análise de eficácia do algoritmo de Naive Bayes

print("\n=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))

print("\n=== Palavras mais relevantes (Naive Bayes) ===")
show_top_features(nb, tfidf)



