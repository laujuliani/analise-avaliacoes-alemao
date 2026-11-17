# Análise de Avaliações de Clientes

Painel em Python e Streamlit que classifica avaliações de clientes em **positivo, neutro ou negativo** e mostra os resultados por filial, por nota e em uma lista de comentários filtrável.

> **Status: protótipo em desenvolvimento.** Hoje funciona com um CSV enviado pelo usuário e com dados fictícios de demonstração. A integração com avaliações reais e a comparação com modelos baseados em Transformers estão no roadmap abaixo.

## Como funciona

1. **Leitura do CSV:** aceita separador `,` ou `;` e codificação UTF-8 ou Latin-1, e localiza automaticamente as colunas de comentário, filial e nota.
2. **Limpeza de texto:** conservadora, preservando acentos e negações ("não gostei"), que são importantes para o sentido.
3. **Rótulos de treino:** vêm da coluna `sentimento` (se existir) ou são aproximados a partir da nota (1–2 negativo, 3 neutro, 4–5 positivo).
4. **Modelo:** TF-IDF (unigramas e bigramas) + Regressão Logística, com `scikit-learn`, usado como **linha de base**.
5. **Avaliação:** os comentários rotulados são divididos em treino e teste (estratificado, semente fixa). O desempenho é medido **somente no conjunto de teste**, com acurácia, F1 macro, relatório por classe e matriz de confusão (aba "Desempenho do modelo").
6. **Painel:** indicadores, distribuição de sentimentos, sentimento por filial e por nota, lista de comentários e exportação dos resultados em CSV.

## Limitações atuais

- Os dados de demonstração são **fictícios** (30 frases) e servem só para mostrar a interface. Com tão poucos exemplos, as métricas são instáveis, e o painel avisa isso quando o conjunto de teste tem menos de 30 comentários.
- Rótulos derivados de notas são uma aproximação: uma nota alta pode acompanhar um comentário crítico, e vice-versa.
- A linha de base não captura bem ironia, contexto e frases mistas.
- Ainda **não há integração** com o Google: o projeto lê um CSV genérico de avaliações.

## Roadmap

- [x] Linha de base com TF-IDF + Regressão Logística
- [x] Separação treino/teste e métricas (acurácia, F1, matriz de confusão)
- [x] Testes automatizados do pré-processamento e do modelo
- [ ] Conectar a avaliações reais da unidade (exportação ou API do Google Business Profile, conforme o acesso disponível)
- [ ] Rotular uma amostra real de comentários para avaliação confiável
- [ ] Validação cruzada e ajuste de hiperparâmetros
- [ ] Fine-tuning de um modelo Transformer em português (BERTimbau) e comparação com a linha de base
- [ ] Extração de tópicos recorrentes (atendimento, preço, entrega, estoque)
- [ ] Resumo automático das avaliações com LLM

## Tecnologias

Python · Streamlit · pandas · scikit-learn · matplotlib

## Como executar

```bash
git clone https://github.com/laujuliani/analise-avaliacoes-alemao.git
cd analise-avaliacoes-alemao

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
python -m streamlit run app.py
```

Sem arquivo, marque "Usar dados fictícios de demonstração" na barra lateral.

### Formato do CSV

| Coluna | Obrigatória | Exemplos de nome aceitos |
| --- | --- | --- |
| Comentário | Sim | `avaliacao`, `comentario`, `review`, `texto` |
| Rótulo de sentimento | Não* | `sentimento`, `label`, `polaridade` (valores: positivo, neutro, negativo) |
| Nota (1 a 5) | Não* | `nota`, `rating`, `estrelas` |
| Filial | Não | `filial`, `loja`, `unidade` |

\* Para treinar e avaliar o modelo é preciso ter o rótulo **ou** a nota, em ao menos 6 comentários e 2 classes (para avaliar, ao menos 2 exemplos por classe).

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Estrutura

```
analise-avaliacoes-alemao/
├── app.py                 # Interface Streamlit
├── src/
│   ├── preprocessing.py   # Leitura, limpeza e montagem de rótulos
│   ├── sentiment.py       # Treino, previsão e avaliação do modelo
│   ├── analysis.py        # Resumos usados no painel
│   └── demo_data.py       # Avaliações fictícias de demonstração
├── tests/                 # Testes automatizados
├── data/                  # Dados locais (CSV/XLSX ignorados pelo Git)
├── notebooks/             # Experimentos
├── requirements.txt
└── requirements-dev.txt
```


