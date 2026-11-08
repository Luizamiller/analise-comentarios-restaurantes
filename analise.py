"""
Análise exploratória e extração de temas dos comentários.

- Temas: dicionário de palavras-chave (regras transparentes, fácil de auditar).
  É a linha de base que um classificador com LLM teria que superar.
- Gráficos: matplotlib, salvos em figures/.
- Insights: números-chave calculados e salvos em reports/insights.json.

Uso:  python src/analise.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data" / "processed" / "avaliacoes_limpas.csv"
FIG = RAIZ / "figures"
REL = RAIZ / "reports"

# ---------------- temas (regex sobre o texto normalizado) ----------------
TEMAS = {
    # Cuidado com falsos positivos: "sem vazar", "preço justo" e "porção generosa"
    # são elogios — por isso as regras de crítica usam termos específicos.
    "Atraso":           r"demor|atras|eternidade",
    "Comida fria":      r"\bfri[oa]\b|gelad|murch",
    "Pedido errado":    r"errad|faltou|esquec|veio com cebola",
    "Embalagem":        r"rasgad|vazou|revirad",
    "Preço":            r"\bcar[oa]\b|preco salgado|taxa de entrega absurda",
    "Quantidade":       r"minuscul|pouca comida",
    "Sabor (crítica)":  r"sem sabor|\bcru[a]?\b|muito salgado|requentad",
    "Sabor (elogio)":   r"saboros|sabor incrivel|delicia|tempero perfeito|melhor",
    "Entrega (elogio)": r"rapido|antes do previsto|educad|quentinho",
}
TEMAS_NEGATIVOS = ["Atraso", "Comida fria", "Pedido errado", "Embalagem",
                   "Preço", "Quantidade", "Sabor (crítica)"]
ORDEM_FAIXAS = ["até 30", "31–40", "41–50", "51–60", "60+"]
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

# ---------------- estilo dos gráficos ----------------
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#b9b8b3"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": CINZA,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def carregar():
    df = pd.read_csv(DADOS, parse_dates=["data_pedido"])
    df["nota"] = df["nota"].astype("Int64")
    # o CSV não guarda a ordem das categorias; restaura para os gráficos
    df["faixa_entrega"] = pd.Categorical(df["faixa_entrega"], ORDEM_FAIXAS, ordered=True)
    df["sentimento_nota"] = pd.Categorical(df["sentimento_nota"],
                                           ["negativa", "neutra", "positiva"], ordered=True)
    return df


def marcar_temas(df):
    df = df.copy()
    texto = df["comentario_norm"].fillna("")
    for tema, padrao in TEMAS.items():
        df[f"tema_{tema}"] = texto.str.contains(padrao, regex=True)
    return df


def _legenda_fonte(ax, texto="Fonte: base sintética de avaliações (n=6.000)"):
    ax.annotate(texto, xy=(0, -0.27), xycoords="axes fraction", fontsize=8, color=TINTA_2)


# ---------------- gráficos ----------------
def fig_distribuicao_notas(df):
    cont = df["nota"].value_counts().sort_index()
    pct = cont / cont.sum() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cores = [LARANJA if n <= 2 else CINZA if n == 3 else AZUL for n in cont.index]
    ax.bar(cont.index.astype(str), pct, color=cores, width=0.62)
    for x, p in zip(cont.index.astype(str), pct):
        ax.text(x, p + 0.8, f"{p:.0f}%", ha="center", color=TINTA, fontsize=9)
    ax.set_title("Distribuição das notas")
    ax.set_ylabel("% das avaliações"); ax.set_xlabel("Nota (estrelas)")
    ax.grid(axis="x", visible=False)
    _legenda_fonte(ax)
    fig.savefig(FIG / "01_distribuicao_notas.png"); plt.close(fig)


def fig_tempo_vs_nota(df):
    g = (df.dropna(subset=["faixa_entrega", "nota"])
           .groupby("faixa_entrega", observed=True)
           .agg(nota_media=("nota", "mean"),
                pct_neg=("nota", lambda s: (s <= 2).mean() * 100),
                n=("nota", "size")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(g.index.astype(str), g["nota_media"], color=AZUL, lw=2, marker="o", ms=7)
    for x, y in zip(g.index.astype(str), g["nota_media"]):
        axes[0].text(x, y + 0.06, f"{y:.2f}", ha="center", fontsize=9, color=TINTA)
    axes[0].set_title("Nota média por tempo de entrega")
    axes[0].set_xlabel("Tempo de entrega (min)"); axes[0].set_ylabel("Nota média")
    axes[0].set_ylim(2.5, 4.8)
    axes[1].bar(g.index.astype(str), g["pct_neg"], color=LARANJA, width=0.62)
    for x, y in zip(g.index.astype(str), g["pct_neg"]):
        axes[1].text(x, y + 0.8, f"{y:.0f}%", ha="center", fontsize=9, color=TINTA)
    axes[1].set_title("% de avaliações negativas (1–2★)")
    axes[1].set_xlabel("Tempo de entrega (min)"); axes[1].grid(axis="x", visible=False)
    _legenda_fonte(axes[0])
    fig.tight_layout(); fig.savefig(FIG / "02_tempo_entrega_vs_nota.png"); plt.close(fig)
    return g


def fig_categorias(df):
    g = (df.groupby("categoria")
           .agg(nota_media=("nota", "mean"), tempo_medio=("tempo_entrega_min", "mean"),
                n=("nota", "size"))
           .sort_values("nota_media"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    axes[0].barh(g.index, g["nota_media"], color=AZUL, height=0.6)
    axes[0].set_xlim(3, 4.4); axes[0].set_title("Nota média por categoria")
    for y, v in enumerate(g["nota_media"]):
        axes[0].text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=TINTA)
    axes[1].barh(g.index, g["tempo_medio"], color=CINZA, height=0.6)
    axes[1].set_title("Tempo médio de entrega (min)")
    for y, v in enumerate(g["tempo_medio"]):
        axes[1].text(v + 0.5, y, f"{v:.0f}", va="center", fontsize=9, color=TINTA)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "03_categorias.png"); plt.close(fig)
    return g


def fig_temas_negativos(df):
    neg = df[(df["nota"] <= 2) & df["tem_comentario"]]
    pct = pd.Series({t: neg[f"tema_{t}"].mean() * 100 for t in TEMAS_NEGATIVOS}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(pct.index, pct.values, color=LARANJA, height=0.6)
    for y, v in enumerate(pct.values):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=9, color=TINTA)
    ax.set_title("O que aparece nas avaliações negativas (1–2★)")
    ax.set_xlabel(f"% dos comentários negativos que citam o tema (n={len(neg)})")
    ax.grid(axis="y", visible=False)
    _legenda_fonte(ax, "Um comentário pode citar mais de um tema.")
    fig.savefig(FIG / "04_temas_negativos.png"); plt.close(fig)
    return pct.sort_values(ascending=False)


def fig_heatmap_pico(df):
    base = df.dropna(subset=["hora"])
    pivo = (base.pivot_table(index="dia_semana", columns="hora", values="tempo_entrega_min",
                             aggfunc="mean").reindex(range(7)))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    im = ax.imshow(pivo.values, cmap="Blues", aspect="auto")
    ax.set_yticks(range(7), DIAS)
    ax.set_xticks(range(pivo.shape[1]), [f"{int(h)}h" for h in pivo.columns])
    ax.grid(False)
    for i in range(pivo.shape[0]):
        for j in range(pivo.shape[1]):
            v = pivo.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > pivo.values.mean() + 3 else TINTA)
    ax.set_title("Tempo médio de entrega (min) por dia e horário")
    fig.colorbar(im, ax=ax, shrink=0.8, label="min")
    fig.savefig(FIG / "05_heatmap_pico.png"); plt.close(fig)
    return pivo


def fig_ranking_restaurantes(df, minimo=60):
    g = (df.groupby("restaurante")
           .agg(n=("nota", "size"), nota=("nota", "mean"),
                pct_neg=("nota", lambda s: (s <= 2).mean() * 100),
                atraso=("tema_Atraso", "mean"))
           .query("n >= @minimo").sort_values("pct_neg", ascending=False))
    top = g.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ax.barh(top.index, top["pct_neg"], color=LARANJA, height=0.6)
    for y, (v, n) in enumerate(zip(top["pct_neg"], top["n"])):
        ax.text(v + 0.5, y, f"{v:.0f}%  (n={n})", va="center", fontsize=8.5, color=TINTA)
    ax.set_title("Restaurantes com maior % de avaliações negativas")
    ax.set_xlabel(f"% de notas 1–2★ (restaurantes com ≥{minimo} avaliações)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "06_ranking_restaurantes.png"); plt.close(fig)
    return g


def dores_por_restaurante(df, restaurantes):
    """Para cada restaurante, o tema negativo mais citado nas notas 1–3."""
    sub = df[df["restaurante"].isin(restaurantes) & (df["nota"] <= 3)]
    cols = [f"tema_{t}" for t in TEMAS_NEGATIVOS]
    pct = sub.groupby("restaurante")[cols].mean() * 100
    pct.columns = TEMAS_NEGATIVOS
    return pct.round(0)


# ---------------- execução ----------------
def main():
    FIG.mkdir(exist_ok=True); REL.mkdir(exist_ok=True)
    df = marcar_temas(carregar())

    fig_distribuicao_notas(df)
    tempo = fig_tempo_vs_nota(df)
    cats = fig_categorias(df)
    temas = fig_temas_negativos(df)
    pico = fig_heatmap_pico(df)
    rank = fig_ranking_restaurantes(df)

    com_nota = df.dropna(subset=["nota"])
    fds = com_nota.groupby("fim_de_semana")["tempo_entrega_min"].mean()
    noite = df["hora"].between(19, 21)
    elogio_c_ressalva = ((df["nota"] == 4) & df[[f"tema_{t}" for t in TEMAS_NEGATIVOS]].any(axis=1)).sum()

    insights = {
        "n_avaliacoes": int(len(df)),
        "nota_media": round(float(com_nota["nota"].mean()), 2),
        "pct_negativas": round(float((com_nota["nota"] <= 2).mean() * 100), 1),
        "pct_com_comentario": round(float(df["tem_comentario"].mean() * 100), 1),
        "nota_ate30": round(float(tempo.loc["até 30", "nota_media"]), 2),
        "nota_60mais": round(float(tempo.loc["60+", "nota_media"]), 2),
        "pct_neg_ate30": round(float(tempo.loc["até 30", "pct_neg"]), 1),
        "pct_neg_60mais": round(float(tempo.loc["60+", "pct_neg"]), 1),
        "corr_tempo_nota": round(float(com_nota[["tempo_entrega_min", "nota"]].astype(float).corr().iloc[0, 1]), 2),
        "temas_negativos_pct": {k: round(float(v), 1) for k, v in temas.items()},
        "tempo_medio_fds": round(float(fds.get(True)), 1),
        "tempo_medio_semana": round(float(fds.get(False)), 1),
        "tempo_medio_19a21h": round(float(df.loc[noite, "tempo_entrega_min"].mean()), 1),
        "tempo_medio_outros_horarios": round(float(df.loc[df["hora"].notna() & ~noite, "tempo_entrega_min"].mean()), 1),
        "nota_por_categoria": cats["nota_media"].round(2).to_dict(),
        "notas_4_com_ressalva": int(elogio_c_ressalva),
        "pct_notas_4_com_ressalva": round(float(elogio_c_ressalva / (df["nota"] == 4).sum() * 100), 1),
        "top_restaurantes_negativos": rank.head(5)["pct_neg"].round(1).to_dict(),
    }
    dores = dores_por_restaurante(df, rank.head(5).index)
    dores.to_csv(REL / "dores_top5_restaurantes.csv")
    (REL / "insights.json").write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(insights, ensure_ascii=False, indent=2))
    print(dores.to_string())


if __name__ == "__main__":
    main()
