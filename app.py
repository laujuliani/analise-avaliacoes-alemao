import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis import NOT_CLASSIFIED, branch_crosstab, sentiment_summary
from src.demo_data import demo_dataframe
from src.preprocessing import (
    BRANCH_COLUMNS,
    RATING_COLUMNS,
    build_training_labels,
    clean_text,
    find_first_column,
    find_text_column,
    normalize_columns,
    read_csv_flexible,
)
from src.sentiment import InsufficientDataError, evaluate_model, predict_sentiments, train_model

st.set_page_config(page_title="Análise de Avaliações", page_icon="💬", layout="wide")

st.title("Análise de Avaliações de Clientes")
st.caption(
    "Protótipo de classificação de sentimento (TF-IDF + Regressão Logística). "
)

COLORS = {"positivo": "#2e7d32", "neutro": "#9e9e9e", "negativo": "#c62828", NOT_CLASSIFIED: "#bdbdbd"}


@st.cache_data
def load_demo() -> pd.DataFrame:
    return demo_dataframe()

st.sidebar.header("Dados")
uploaded = st.sidebar.file_uploader("Envie um CSV de avaliações", type=["csv"])
use_demo = st.sidebar.checkbox("Usar dados fictícios de demonstração", value=uploaded is None)

if uploaded is not None:
    try:
        raw_df = read_csv_flexible(uploaded)
    except Exception as exc:  # arquivo ilegível
        st.error(f"Não foi possível ler o CSV: {exc}")
        st.stop()
elif use_demo:
    raw_df = load_demo()
else:
    st.info("Envie um CSV pela barra lateral ou ative os dados de demonstração.")
    st.stop()

df = normalize_columns(raw_df)
text_col = find_text_column(df)
if text_col is None:
    st.error("Não encontrei a coluna de comentários. Use um nome como 'avaliacao', 'comentario' ou 'review'.")
    st.stop()

df = df.dropna(subset=[text_col]).copy()
df["texto_limpo"] = df[text_col].map(clean_text)
df = df[df["texto_limpo"] != ""].copy()
if df.empty:
    st.warning("Nenhum comentário válido foi encontrado no arquivo.")
    st.stop()

# rótulo + modelo
labels, label_source = build_training_labels(df)
df["sentimento_informado"] = labels
labeled = df[labels.notna()]
n_classes = labeled["sentimento_informado"].nunique()

metrics, metrics_msg = None, None
if len(labeled) >= 6 and n_classes >= 2:
    try:
        metrics = evaluate_model(labeled["texto_limpo"], labeled["sentimento_informado"])
    except InsufficientDataError as exc:
        metrics_msg = str(exc)
    model = train_model(labeled["texto_limpo"], labeled["sentimento_informado"])
    df["sentimento_previsto"] = predict_sentiments(model, df["texto_limpo"])
else:
    df["sentimento_previsto"] = NOT_CLASSIFIED
    st.warning(
        "Não há rótulos suficientes para treinar o modelo. Inclua uma coluna 'sentimento' "
        "(positivo/neutro/negativo) ou 'nota' (1 a 5), com ao menos 6 comentários e 2 classes."
    )

st.sidebar.header("Filtros")
branch_col = find_first_column(df, BRANCH_COLUMNS)
rating_col = find_first_column(df, RATING_COLUMNS)

filtered = df
if branch_col:
    branches = sorted(df[branch_col].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect("Filial", branches, default=branches)
    filtered = filtered[filtered[branch_col].astype(str).isin(chosen)]

options = ["Todos"] + [s for s in ["positivo", "neutro", "negativo", NOT_CLASSIFIED]
                       if s in set(df["sentimento_previsto"])]
sentiment_filter = st.sidebar.selectbox("Sentimento", options)
if sentiment_filter != "Todos":
    filtered = filtered[filtered["sentimento_previsto"] == sentiment_filter]

counts = df["sentimento_previsto"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avaliações", len(df))
c2.metric("Positivas", int(counts.get("positivo", 0)))
c3.metric("Neutras", int(counts.get("neutro", 0)))
c4.metric("Negativas", int(counts.get("negativo", 0)))

if label_source:
    st.caption(f"Origem dos rótulos de treino: {label_source}.")

tab_overview, tab_comments, tab_model = st.tabs(["Visão geral", "Comentários", "Desempenho do modelo"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Distribuição de sentimentos")
        summary = sentiment_summary(df)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(summary.index, summary.values, color=[COLORS.get(s, "#607d8b") for s in summary.index])
        ax.set_ylabel("Avaliações")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        if branch_col:
            st.subheader("Sentimento por filial")
            st.dataframe(branch_crosstab(df, branch_col))
        else:
            st.info("O CSV não tem coluna de filial (filial, loja ou unidade).")
    if rating_col:
        st.subheader("Sentimento previsto por nota")
        st.dataframe(pd.crosstab(df[rating_col], df["sentimento_previsto"]))

with tab_comments:
    st.subheader("Comentários classificados")
    shown = [c for c in [text_col, "sentimento_previsto", "sentimento_informado", branch_col, rating_col] if c]
    st.dataframe(filtered[shown], hide_index=True)
    st.download_button(
        "Baixar resultados (CSV)",
        filtered.drop(columns=["texto_limpo"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="avaliacoes_classificadas.csv",
        mime="text/csv",
    )

with tab_model:
    st.subheader("Desempenho em dados que o modelo não viu")
    st.write(
        "Parte dos comentários rotulados é separada **antes** do treino e usada só para medir o modelo. "
        "Prever os mesmos comentários usados no treino não mede desempenho, só memorização."
    )
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Acurácia", f"{metrics['accuracy']:.0%}")
        m2.metric("F1 macro", f"{metrics['f1_macro']:.2f}")
        m3.metric("Treino", metrics["n_train"])
        m4.metric("Teste", metrics["n_test"])
        if not metrics["reliable"]:
            st.warning(
                f"O conjunto de teste tem só {metrics['n_test']} comentários, então estas métricas são "
                "muito instáveis. Use dados reais, com centenas de exemplos, antes de tirar conclusões."
            )
        st.markdown("**Relatório por classe**")
        st.dataframe(metrics["report"].round(2))
        st.markdown("**Matriz de confusão** (linhas: valor real; colunas: previsão)")
        st.dataframe(metrics["confusion"])
    elif metrics_msg:
        st.info(f"Não foi possível avaliar o modelo: {metrics_msg}")
    else:
        st.info("Sem rótulos suficientes para avaliar o modelo.")
