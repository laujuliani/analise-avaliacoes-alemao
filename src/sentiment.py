# Classificador de sentimento (linha de base)
import math

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import SENTIMENT_ORDER

# Abaixo disso, as métricas do conjunto de teste são muito instáveis.
MIN_TEST_SIZE_RELIABLE = 30


class InsufficientDataError(ValueError):
    """Dados rotulados insuficientes para separar treino e teste."""


def train_model(texts, labels):
    #Treina um classificador simples (TF-IDF + Regressão Logística)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(texts, labels)
    return model


def predict_sentiments(model, texts):
    return model.predict(texts)


def evaluate_model(texts, labels, test_size=0.25, seed=42):
    # Mede o desempenho em um conjunto de teste separado, que o modelo nunca viu.
    # InsufficientDataError se não houver exemplos suficientes.

    texts = pd.Series(list(texts)).reset_index(drop=True)
    labels = pd.Series(list(labels)).reset_index(drop=True)

    counts = labels.value_counts()
    n_classes = len(counts)
    if n_classes < 2:
        raise InsufficientDataError("É preciso ter ao menos duas classes de sentimento para avaliar o modelo.")
    if counts.min() < 2:
        rare = counts.idxmin()
        raise InsufficientDataError(
            f"A classe '{rare}' tem só {counts.min()} exemplo(s); são necessários ao menos 2 por classe."
        )

    n = len(texts)
    n_test = max(math.ceil(test_size * n), n_classes)
    if n - n_test < n_classes:
        raise InsufficientDataError("Poucos exemplos rotulados para separar treino e teste.")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=n_test, random_state=seed, stratify=labels
    )
    model = train_model(x_train, y_train)
    pred = model.predict(x_test)

    classes = [c for c in SENTIMENT_ORDER if c in set(labels)]
    report = pd.DataFrame(
        classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    ).T
    confusion = pd.DataFrame(
        confusion_matrix(y_test, pred, labels=classes),
        index=[f"real: {c}" for c in classes],
        columns=[f"previsto: {c}" for c in classes],
    )
    return {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, labels=classes, average="macro", zero_division=0),
        "report": report,
        "confusion": confusion,
        "reliable": len(x_test) >= MIN_TEST_SIZE_RELIABLE,
    }
