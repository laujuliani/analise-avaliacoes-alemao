import io

import pandas as pd

from src.preprocessing import (
    build_training_labels,
    clean_text,
    find_text_column,
    normalize_columns,
    read_csv_flexible,
)


def test_clean_text_keeps_accents_and_negation():
    assert clean_text("  NÃO gostei!  Veja https://x.com/a ") == "não gostei! veja"


def test_normalize_columns_removes_accents():
    df = normalize_columns(pd.DataFrame({"Avaliação": ["a"], "Nota Geral": [5]}))
    assert list(df.columns) == ["avaliacao", "nota_geral"]


def test_find_text_column():
    assert find_text_column(pd.DataFrame({"id": [1], "comentario": ["ok"]})) == "comentario"
    assert find_text_column(pd.DataFrame({"id": [1]})) is None


def test_labels_from_ratings():
    df = pd.DataFrame({"nota": [1, 2, 3, 4, 5, None]})
    labels, source = build_training_labels(df)
    assert list(labels[:5]) == ["negativo", "negativo", "neutro", "positivo", "positivo"]
    assert pd.isna(labels.iloc[5])
    assert "notas" in source


def test_labels_from_column_take_priority_and_ignore_invalid():
    df = pd.DataFrame({"sentimento": ["Positivo", "NEG", "talvez"], "nota": [1, 1, 1]})
    labels, source = build_training_labels(df)
    assert list(labels[:2]) == ["positivo", "negativo"]
    assert pd.isna(labels.iloc[2])
    assert "informados" in source


def test_no_label_columns():
    labels, source = build_training_labels(pd.DataFrame({"avaliacao": ["a", "b"]}))
    assert source is None and labels.isna().all()


def test_read_csv_semicolon_latin1():
    data = "avaliacao;nota\nótimo atendimento;5\n".encode("latin-1")
    df = read_csv_flexible(io.BytesIO(data))
    assert list(df.columns) == ["avaliacao", "nota"]
    assert df.loc[0, "avaliacao"] == "ótimo atendimento"
