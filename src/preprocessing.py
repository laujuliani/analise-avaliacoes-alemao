# Leitura, padronização e limpeza dos dados de avaliações.
import re
import unicodedata

import pandas as pd

TEXT_CANDIDATES = [
    "avaliacao", "avaliação", "comentario", "comentário",
    "review", "texto", "text", "comment", "resenha",
]
LABEL_COLUMNS = ["sentimento", "label", "classe", "polaridade"]
RATING_COLUMNS = ["nota", "rating", "estrelas", "avaliacao_nota"]
BRANCH_COLUMNS = ["filial", "loja", "unidade", "branch"]

SENTIMENT_ORDER = ["positivo", "neutro", "negativo"]

LABEL_MAP = {
    "positive": "positivo", "pos": "positivo", "1": "positivo",
    "negative": "negativo", "neg": "negativo", "-1": "negativo",
    "neutral": "neutro", "neu": "neutro", "0": "neutro",
}


def _slug(name) -> str:
    # Minúsculas, sem acento e com '_' no lugar de espaços
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return ascii_name.strip().lower().replace(" ", "_")


def read_csv_flexible(source) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Padroniza nomes de colunas sem alterar o conteúdo.
    result = df.copy()
    result.columns = [_slug(col) for col in result.columns]
    return result


def find_first_column(df: pd.DataFrame, candidates):
    # Devolve o primeiro nome de coluna de `candidates` presente em `df`
    return next((c for c in candidates if c in df.columns), None)


def find_text_column(df: pd.DataFrame):
    for candidate in TEXT_CANDIDATES:
        slug = _slug(candidate)
        if slug in df.columns:
            return slug
    for col in df.columns:
        if any(term in col for term in ["coment", "review", "texto", "text"]):
            return col
    return None


def clean_text(value) -> str:
    # Limpeza: preserva acentos, palavras e negações.
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rating_to_label(value):
   # Nota 1-5 -> sentimento aproximado (1-2 negativo, 3 neutro, 4-5 positivo)
    if pd.isna(value):
        return None
    if value <= 2:
        return "negativo"
    if value == 3:
        return "neutro"
    if value >= 4:
        return "positivo"
    return None


def build_training_labels(df: pd.DataFrame):
    # Monta os rótulos de treino a partir da coluna de sentimento ou das notas.

    # Retorna (rótulos, descrição da origem). Linhas sem rótulo válido ficam com None.
    # Se não houver nenhuma das duas colunas, devolve (série vazia de None, None).
    
    label_col = find_first_column(df, LABEL_COLUMNS)
    rating_col = find_first_column(df, RATING_COLUMNS)

    if label_col:
        raw = df[label_col].astype(str).str.lower().str.strip()
        labels = raw.map(lambda x: LABEL_MAP.get(x, x))
        labels = labels.where(labels.isin(SENTIMENT_ORDER), None)
        return labels, "rótulos informados no CSV"

    if rating_col:
        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        labels = ratings.map(rating_to_label)
        return labels, "rótulos aproximados derivados das notas (1–2 negativo, 3 neutro, 4–5 positivo)"

    return pd.Series([None] * len(df), index=df.index, dtype=object), None
