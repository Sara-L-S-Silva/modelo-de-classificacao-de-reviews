import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
### NOVO: Importando a Matriz de Confusão e SMOTE ###
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
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

# Carrega e cria o rótulo (y)
csv_file_path = os.path.join(path, "olist_order_reviews_dataset.csv")
df = pd.read_csv(csv_file_path)

def score_to_label(score):
    if score >= 4:
        return "positivo"
    elif score <= 2:
        return "negativo"
    else:
        return None

df["label"] = df["review_score"].apply(score_to_label)
df = df.dropna(subset=["label"])

# Diagnóstico do Desbalanceamento #
# Vamos verificar a proporção de cada classe.
print("\n### Proporção das Classes ###")
print(df['label'].value_counts(normalize=True))
print("--------------------------------\n")
# Se a classe 'negativo' for < 20-30%, o desbalanceamento é significativo.

# Concatenar título + mensagem e pré-processar
df["review_text"] = (df["review_comment_title"].fillna("") + " " +
                      df["review_comment_message"].fillna(""))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_pt]
    return " ".join(tokens)

df["clean_text"] = df["review_text"].apply(clean_text)

# Separa o treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Vetorização com TF-IDF
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=5)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# MODELOS 

#TÉCNICA 1: USANDO class_weight='balanced' 
print("\n=== Regressão Logística com 'class_weight' ===")
logreg_balanced = LogisticRegression(max_iter=1000, C=0.5, penalty="l2", class_weight='balanced')
logreg_balanced.fit(X_train_tfidf, y_train)
y_pred_logreg_balanced = logreg_balanced.predict(X_test_tfidf)

print(classification_report(y_test, y_pred_logreg_balanced))

# Ferramenta de Análise - Matriz de Confusão
# A matriz de confusão ajuda a visualizar os erros.
# Linhas: Realidade | Colunas: Previsão do Modelo
cm = confusion_matrix(y_test, y_pred_logreg_balanced, labels=logreg_balanced.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg_balanced.classes_)
disp.plot()
plt.title("Matriz de Confusão - Regressão Logística com 'class_weight'")
plt.show()


#TÉCNICA 2: Balanceamento com SMOTE (Synthetic Minority Over-sampling Technique) 
# SMOTE cria novos exemplos sintéticos da classe minoritária (negativo) para balancear o treino.
print("\n=== Regressão Logística com SMOTE ===")
smote = SMOTE(random_state=42)
# IMPORTANTE: SMOTE só é aplicado nos dados de TREINO!
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print("Tamanho do treino antes do SMOTE:", X_train_tfidf.shape)
print("Tamanho do treino depois do SMOTE:", X_train_smote.shape)

logreg_smote = LogisticRegression(max_iter=1000, C=0.5, penalty="l2")
logreg_smote.fit(X_train_smote, y_train_smote)
y_pred_logreg_smote = logreg_smote.predict(X_test_tfidf)

print(classification_report(y_test, y_pred_logreg_smote))

# Matriz de confusão para o modelo com SMOTE
cm_smote = confusion_matrix(y_test, y_pred_logreg_smote, labels=logreg_smote.classes_)
disp_smote = ConfusionMatrixDisplay(confusion_matrix=cm_smote, display_labels=logreg_smote.classes_)
disp_smote.plot()
plt.title("Matriz de Confusão - Regressão Logística com SMOTE")
plt.show()