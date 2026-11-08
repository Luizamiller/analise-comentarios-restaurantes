"""
(Opcional) Classificação de temas com LLM — comparação com as regras.

As regras de palavras-chave em `analise.py` funcionam bem quando o vocabulário é
previsível. Em comentários reais aparecem gírias, ironia e erros de digitação
("demorou uma vida", "veio um gelo", "nunca mais peço"). Este script envia uma
amostra de comentários para um LLM, pede os temas em JSON e mede a concordância
com as regras.

Requisitos:
    pip install anthropic
    export ANTHROPIC_API_KEY=...          # sua chave
    export MODELO_LLM=claude-haiku-5-5    # opcional

Uso:  python src/temas_llm.py --amostra 100
Saída: reports/comparacao_regras_vs_llm.csv
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from analise import TEMAS_NEGATIVOS, carregar, marcar_temas

RAIZ = Path(__file__).resolve().parents[1]
MODELO = os.getenv("MODELO_LLM", "claude-haiku-5-5")

PROMPT = """Você classifica avaliações de restaurantes de delivery.
Temas possíveis (apenas críticas): {temas}.
Responda SOMENTE com um JSON no formato {{"temas": [...]}} contendo os temas
criticados no comentário. Se não houver crítica, devolva {{"temas": []}}.

Comentário: "{comentario}"
"""


def classificar(cliente, comentario: str) -> list[str]:
    resp = cliente.messages.create(
        model=MODELO, max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(
            temas=", ".join(TEMAS_NEGATIVOS), comentario=comentario)}],
    )
    texto = resp.content[0].text.strip()
    try:
        temas = json.loads(texto[texto.find("{"): texto.rfind("}") + 1])["temas"]
    except (ValueError, KeyError):
        return []
    return [t for t in temas if t in TEMAS_NEGATIVOS]


def main():
    import anthropic  # importado aqui para o resto do projeto não depender dele

    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", type=int, default=100)
    args = ap.parse_args()

    df = marcar_temas(carregar())
    amostra = (df[df["tem_comentario"] & (df["nota"] <= 3)]
               .sample(args.amostra, random_state=7))
    cliente = anthropic.Anthropic()

    linhas = []
    for _, r in amostra.iterrows():
        llm = set(classificar(cliente, r["comentario"]))
        regras = {t for t in TEMAS_NEGATIVOS if r[f"tema_{t}"]}
        linhas.append({"comentario": r["comentario"], "nota": r["nota"],
                       "regras": sorted(regras), "llm": sorted(llm),
                       "concorda": llm == regras})
    res = pd.DataFrame(linhas)
    saida = RAIZ / "reports" / "comparacao_regras_vs_llm.csv"
    res.to_csv(saida, index=False)
    print(f"Concordância exata regras x LLM: {res['concorda'].mean():.0%}")
    print(f"Divergências salvas em {saida.relative_to(RAIZ)} para revisão manual.")


if __name__ == "__main__":
    main()
