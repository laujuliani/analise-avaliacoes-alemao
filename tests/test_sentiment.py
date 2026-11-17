import pandas as pd
import pytest

from src.analysis import branch_crosstab, sentiment_summary
from src.demo_data import demo_dataframe
from src.sentiment import InsufficientDataError, evaluate_model, predict_sentiments, train_model


def test_evaluate_uses_separate_test_set():
    df = demo_dataframe()
    result = evaluate_model(df["avaliacao"], df["sentimento"])
    assert result["n_train"] + result["n_test"] == len(df)
    assert result["n_test"] >= 3
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["confusion"].to_numpy().sum() == result["n_test"]
    assert result["reliable"] is False  # só 30 exemplos fictícios


def test_evaluate_is_reproducible():
    df = demo_dataframe()
    a = evaluate_model(df["avaliacao"], df["sentimento"])
    b = evaluate_model(df["avaliacao"], df["sentimento"])
    assert a["accuracy"] == b["accuracy"]


def test_evaluate_rejects_single_class():
    with pytest.raises(InsufficientDataError):
        evaluate_model(["a", "b", "c"], ["positivo"] * 3)


def test_evaluate_rejects_rare_class():
    texts = ["a", "b", "c", "d", "e"]
    labels = ["positivo", "positivo", "positivo", "negativo", "negativo"]
    labels[-1] = "neutro"  # neutro e negativo com 1 exemplo cada
    with pytest.raises(InsufficientDataError):
        evaluate_model(texts, labels)


def test_train_and_predict_shape():
    df = demo_dataframe()
    model = train_model(df["avaliacao"], df["sentimento"])
    pred = predict_sentiments(model, ["péssimo atendimento, não recomendo", "excelente, recomendo muito"])
    assert len(pred) == 2


def test_summary_order_and_crosstab():
    df = pd.DataFrame({
        "sentimento_previsto": ["negativo", "positivo", "positivo", "neutro"],
        "filial": ["A", "A", "B", "B"],
    })
    assert list(sentiment_summary(df).index) == ["positivo", "neutro", "negativo"]
    table = branch_crosstab(df, "filial")
    assert table.loc["Total", "Total"] == 4
