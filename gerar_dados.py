"""
Gera uma base SINTÉTICA de avaliações de restaurantes de delivery.

Por que sintética? Avaliações reais de plataformas não são públicas para uso
livre. Aqui simulamos o formato e os problemas típicos de uma base real
(duplicatas, nulos, datas em formatos diferentes, categorias escritas de jeitos
diferentes, outliers de tempo de entrega) para treinar o pipeline de limpeza e
análise. Os padrões embutidos estão documentados no README.

Uso:  python src/gerar_dados.py
Saída: data/raw/avaliacoes_raw.csv
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd

SEED = 42
N = 6000
RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "data" / "raw" / "avaliacoes_raw.csv"

rng = np.random.default_rng(SEED)
random.seed(SEED)

CIDADES = {"São Paulo": 0.38, "Rio de Janeiro": 0.20, "Belo Horizonte": 0.12,
           "Porto Alegre": 0.10, "Curitiba": 0.10, "Pelotas": 0.10}

# categoria: (peso, ticket médio, tempo base de entrega)
CATEGORIAS = {
    "Pizza": (0.22, 75, 42), "Hambúrguer": (0.20, 52, 35),
    "Japonesa": (0.14, 95, 45), "Brasileira": (0.16, 38, 32),
    "Lanches": (0.12, 30, 28), "Saudável": (0.08, 45, 30),
    "Doces & Sobremesas": (0.08, 35, 30),
}
# variações de grafia que aparecem na base "suja"
GRAFIAS = {
    "Pizza": ["Pizza", "pizza", "PIZZA", "Pizzaria"],
    "Hambúrguer": ["Hambúrguer", "hamburguer", "Hamburguer", "burger"],
    "Japonesa": ["Japonesa", "japonesa", "Comida Japonesa", "sushi"],
    "Brasileira": ["Brasileira", "brasileira", "Comida Brasileira"],
    "Lanches": ["Lanches", "lanches", "Lanche"],
    "Saudável": ["Saudável", "saudavel", "Saudavel"],
    "Doces & Sobremesas": ["Doces & Sobremesas", "doces", "Sobremesas"],
}
PREFIXOS = ["Cantina", "Casa", "Empório", "Point", "Cozinha", "Ateliê", "Recanto"]
SUFIXOS = ["da Vila", "do Centro", "Express", "Gourmet", "da Esquina", "Prime", "do Bairro"]

# ---------- banco de frases por tema (positivo / negativo) ----------
POS = {
    "sabor": ["comida muito saborosa", "tempero perfeito", "estava uma delícia",
              "melhor {prato} que já pedi", "sabor incrível"],
    "entrega": ["chegou super rápido", "entrega antes do previsto",
                "entregador muito educado", "chegou quentinho"],
    "embalagem": ["embalagem caprichada", "veio tudo bem embalado, sem vazar"],
    "preco": ["ótimo custo-benefício", "preço justo pela quantidade"],
    "quantidade": ["porção generosa", "dá pra duas pessoas tranquilo"],
}
NEG = {
    "atraso": ["demorou demais", "atrasou mais de {min} minutos",
               "pedido chegou muito atrasado", "esperei uma eternidade"],
    "temperatura": ["chegou frio", "comida gelada", "{prato} chegou fria e murcha"],
    "embalagem": ["embalagem rasgada", "molho vazou todo na sacola",
                  "veio tudo revirado"],
    "pedido_errado": ["veio item errado", "faltou o refrigerante",
                      "pedi sem cebola e veio com cebola", "esqueceram a sobremesa"],
    "preco": ["muito caro pelo que entrega", "preço salgado", "taxa de entrega absurda"],
    "quantidade": ["porção minúscula", "veio pouca comida"],
    "sabor": ["sem sabor nenhum", "{prato} estava cru", "muito salgado",
              "gosto de requentado"],
}
PRATOS = {"Pizza": "pizza", "Hambúrguer": "hambúrguer", "Japonesa": "sushi",
          "Brasileira": "feijoada", "Lanches": "lanche", "Saudável": "salada",
          "Doces & Sobremesas": "bolo"}
CONECTORES = [", ", ". ", " mas ", " e ", "!! ", "... "]


def frase(banco, tema, prato, minutos):
    return random.choice(banco[tema]).format(prato=prato, min=minutos)


def montar_comentario(nota, categoria, minutos_atraso, temas_neg_restaurante):
    prato = PRATOS[categoria]
    partes = []
    if nota >= 4:
        temas = random.sample(list(POS), k=random.choice([1, 2]))
        partes = [frase(POS, t, prato, minutos_atraso) for t in temas]
        if nota == 4 and random.random() < 0.35:   # elogio com ressalva
            t = "atraso" if minutos_atraso > 15 else random.choice(
                [k for k in NEG if k != "atraso"])
            partes.append(frase(NEG, t, prato, minutos_atraso))
    else:
        temas = []
        if minutos_atraso > 12:
            temas.append(random.choice(["atraso", "atraso", "temperatura"]))
        temas += random.sample(temas_neg_restaurante, k=1)
        if nota == 3 and random.random() < 0.5:
            temas.append(random.choice(list(POS)))
            partes.append(frase(POS, temas[-1], prato, minutos_atraso))
            temas = temas[:-1]
        partes = [frase(NEG, t, prato, minutos_atraso) for t in dict.fromkeys(temas)] + partes
    texto = partes[0]
    for p in partes[1:]:
        texto += random.choice(CONECTORES) + p
    if nota >= 4:
        emoji = random.choice(["", "", "", " 😍", " 👍", " 😋"])
    elif nota <= 2:
        emoji = random.choice(["", "", " 😡", " 🙄", " 👎"])
    else:
        emoji = random.choice(["", "", " 🙄", " 😐"])
    return texto.capitalize() + emoji


def main():
    # ----- restaurantes -----
    restaurantes = []
    nomes_usados = set()
    for cat, (peso, ticket, tbase) in CATEGORIAS.items():
        for _ in range(int(peso * 60)):
            while True:
                nome = f"{random.choice(PREFIXOS)} {random.choice(['Sabor', 'Fogo', 'Arte', 'Mesa', 'Brasa', 'Grão', 'Folha'])} {random.choice(SUFIXOS)}"
                if nome not in nomes_usados:
                    nomes_usados.add(nome)
                    break
            qualidade = rng.normal(0, 0.75)  # efeito fixo do restaurante
            # cada restaurante tem 2 "dores" mais frequentes
            dores = random.sample(list(NEG), k=2)
            restaurantes.append(dict(nome=nome, categoria=cat, ticket=ticket,
                                     tbase=tbase, qualidade=qualidade, dores=dores,
                                     cidade=rng.choice(list(CIDADES), p=list(CIDADES.values()))))
    pesos_rest = np.array([CATEGORIAS[r["categoria"]][0] for r in restaurantes])
    pesos_rest = pesos_rest / pesos_rest.sum()

    # ----- datas: jan–set/2026, mais pedidos sex/sáb/dom e à noite -----
    dias = pd.date_range("2026-01-01", "2026-09-30", freq="D")
    peso_dia = np.array([1.6 if d.dayofweek >= 4 else 1.0 for d in dias])
    peso_dia /= peso_dia.sum()

    linhas = []
    for i in range(N):
        r = restaurantes[rng.choice(len(restaurantes), p=pesos_rest)]
        dia = rng.choice(dias, p=peso_dia)
        hora = int(rng.choice([11, 12, 13, 18, 19, 20, 21, 22],
                              p=[.08, .14, .10, .10, .18, .20, .13, .07]))
        quando = pd.Timestamp(dia) + pd.Timedelta(hours=hora, minutes=int(rng.integers(0, 60)))

        # tempo de entrega: pico de jantar e fim de semana atrasam mais
        extra_pico = 8 if hora in (19, 20, 21) else 0
        extra_fds = 5 if quando.dayofweek >= 4 else 0
        tempo = max(10, rng.normal(r["tbase"] + extra_pico + extra_fds, 10))
        prometido = r["tbase"] + 5
        atraso = int(max(0, tempo - prometido))

        # nota depende de qualidade do restaurante + atraso (efeito forte)
        latente = 4.3 + r["qualidade"] - 0.07 * atraso + rng.normal(0, 0.8)
        nota = int(np.clip(round(latente), 1, 5))

        valor = round(max(15, rng.normal(r["ticket"], r["ticket"] * 0.3)), 2)
        comentario = montar_comentario(nota, r["categoria"], atraso, r["dores"])

        linhas.append({
            "id_avaliacao": 100000 + i,
            "data_pedido": quando,
            "restaurante": r["nome"],
            "categoria": random.choice(GRAFIAS[r["categoria"]]),
            "cidade": r["cidade"],
            "nota": nota,
            "tempo_entrega_min": round(tempo),
            "valor_pedido": valor,
            "comentario": comentario,
        })

    df = pd.DataFrame(linhas)

    # ================= sujeira proposital =================
    n = len(df)
    # 1) datas em formatos diferentes
    fmt = rng.choice(["iso", "br", "br_curto"], size=n, p=[.6, .3, .1])
    df["data_pedido"] = [
        d.strftime("%Y-%m-%d %H:%M") if f == "iso"
        else d.strftime("%d/%m/%Y %H:%M") if f == "br"
        else d.strftime("%d/%m/%y")
        for d, f in zip(df["data_pedido"], fmt)
    ]
    # 2) nota como texto, às vezes "4.0", às vezes fora da escala
    df["nota"] = df["nota"].astype(str)
    idx = rng.choice(n, 300, replace=False); df.loc[idx, "nota"] = df.loc[idx, "nota"] + ".0"
    idx = rng.choice(n, 25, replace=False); df.loc[idx, "nota"] = rng.choice(["0", "6", "10"], 25)
    # 3) valor com vírgula decimal e "R$"
    df["valor_pedido"] = df["valor_pedido"].astype(str)
    idx = rng.choice(n, 900, replace=False)
    df.loc[idx, "valor_pedido"] = "R$ " + df.loc[idx, "valor_pedido"].str.replace(".", ",", regex=False)
    # 4) outliers / erros de sistema no tempo de entrega
    idx = rng.choice(n, 40, replace=False); df.loc[idx, "tempo_entrega_min"] = rng.choice([-5, 0, 480, 999], 40)
    # 5) espaços e caixa nos nomes de cidade e restaurante
    idx = rng.choice(n, 500, replace=False); df.loc[idx, "cidade"] = " " + df.loc[idx, "cidade"].str.lower() + "  "
    idx = rng.choice(n, 300, replace=False); df.loc[idx, "restaurante"] = df.loc[idx, "restaurante"].str.upper()
    # 6) nulos
    for col, k in [("comentario", 380), ("tempo_entrega_min", 120), ("cidade", 60), ("valor_pedido", 50)]:
        idx = rng.choice(n, k, replace=False); df.loc[idx, col] = np.nan
    # comentários "vazios" que não são nulos
    idx = rng.choice(n, 90, replace=False); df.loc[idx, "comentario"] = rng.choice(["", " ", ".", "-", "ok"], 90)
    # 7) duplicatas exatas (reenvio da mesma avaliação)
    dups = df.sample(180, random_state=SEED)
    df = pd.concat([df, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"{len(df)} linhas salvas em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
