"""Funções de resumo usadas pelo painel."""
import pandas as pd

from src.preprocessing import SENTIMENT_ORDER

NOT_CLASSIFIED = "não classificado"


def sentiment_summary(df: pd.DataFrame, sentiment_column="sentimento_previsto") -> pd.Series:
    """Contagem por sentimento, sempre na ordem positivo, neutro, negativo."""
    if sentiment_column not in df.columns:
        raise ValueError(f"Coluna ausente: {sentiment_column}")
    counts = df[sentiment_column].value_counts()
    order = [s for s in SENTIMENT_ORDER + [NOT_CLASSIFIED] if s in counts.index]
    extras = [s for s in counts.index if s not in order]
    return counts.reindex(order + extras)


def branch_crosstab(df: pd.DataFrame, branch_column: str, sentiment_column="sentimento_previsto") -> pd.DataFrame:
    """Tabela de sentimentos por filial, com totais."""
    table = pd.crosstab(df[branch_column], df[sentiment_column], margins=True, margins_name="Total")
    ordered = [c for c in SENTIMENT_ORDER + [NOT_CLASSIFIED] if c in table.columns]
    others = [c for c in table.columns if c not in ordered and c != "Total"]
    return table[ordered + others + ["Total"]]
