"""Avaliações FICTÍCIAS, usadas apenas para demonstrar a interface."""
import pandas as pd

_POSITIVOS = [
    "Atendimento excelente, fui muito bem atendido.",
    "Encontrei a peça que precisava e o preço foi justo.",
    "Ótima variedade de peças e equipe atenciosa.",
    "Bom atendimento e entrega dentro do prazo.",
    "Resolveram tudo rapidamente. Muito obrigado!",
    "Vendedor educado e a peça serviu perfeitamente no meu carro.",
    "Loja organizada, achei tudo rápido e o atendimento foi ótimo.",
    "Recomendo! Preço bom e o pessoal entende do assunto.",
    "Fui muito bem recebido e saí com o problema resolvido.",
    "Entrega rápida e a peça chegou bem embalada.",
]
_NEUTROS = [
    "Demoraram para responder no WhatsApp, mas depois resolveram.",
    "Preço um pouco alto, mas fui bem atendido.",
    "A loja é normal, nada de especial.",
    "Atendimento regular, a loja tinha o que eu precisava.",
    "Consegui o que queria, mas esperei bastante na fila.",
    "Peça ok, o preço é parecido com o de outras lojas.",
    "Atendimento razoável, poderia ser mais ágil.",
    "Gostei da peça, mas a entrega atrasou um dia.",
    "Nem bom nem ruim, comprei e foi tudo dentro do esperado.",
    "A equipe é simpática, porém não tinha a peça em estoque.",
]
_NEGATIVOS = [
    "Não gostei do atendimento, ninguém resolveu meu problema.",
    "A peça veio errada e tive dificuldade para trocar.",
    "Péssima experiência, não recomendo.",
    "Esperei muito tempo e ninguém me atendeu.",
    "Peça com defeito e a loja não quis trocar.",
    "Atendimento ruim e vendedor mal-educado.",
    "Entrega atrasou uma semana e ninguém deu satisfação.",
    "Preço abusivo e peça de baixa qualidade.",
    "Pedi um orçamento e nunca me responderam.",
    "Voltei duas vezes e continuaram sem resolver o problema.",
]


def demo_dataframe() -> pd.DataFrame:
    textos = _POSITIVOS + _NEUTROS + _NEGATIVOS
    sentimentos = ["positivo"] * 10 + ["neutro"] * 10 + ["negativo"] * 10
    filiais = ["Campinas" if i % 2 == 0 else "Valinhos" for i in range(len(textos))]
    return pd.DataFrame({"avaliacao": textos, "sentimento": sentimentos, "filial": filiais})
