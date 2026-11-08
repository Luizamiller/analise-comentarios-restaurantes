"""
Limpeza e padronização da base de avaliações.

Cada etapa registra quantas linhas afetou em `log`, para que o relatório de
qualidade de dados seja gerado a partir do próprio pipeline (e não escrito à mão).

Uso:  python src/limpeza.py
Entrada: data/raw/avaliacoes_raw.csv
Saída:   data/processed/avaliacoes_limpas.csv
         reports/qualidade_dados.csv
"""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "raw" / "avaliacoes_raw.csv"
SAIDA = RAIZ / "data" / "processed" / "avaliacoes_limpas.csv"
RELATORIO = RAIZ / "reports" / "qualidade_dados.csv"

# grafia encontrada (minúscula, sem acento) -> categoria padronizada
MAPA_CATEGORIA = {
    "pizza": "Pizza", "pizzaria": "Pizza",
    "hamburguer": "Hambúrguer", "burger": "Hambúrguer",
    "japonesa": "Japonesa", "comida japonesa": "Japonesa", "sushi": "Japonesa",
    "brasileira": "Brasileira", "comida brasileira": "Brasileira",
    "lanches": "Lanches", "lanche": "Lanches",
    "saudavel": "Saudável",
    "doces & sobremesas": "Doces & Sobremesas", "doces": "Doces & Sobremesas",
    "sobremesas": "Doces & Sobremesas",
}
COMENTARIOS_VAZIOS = {"", ".", "-", "ok"}
TEMPO_MIN, TEMPO_MAX = 5, 180          # minutos plausíveis de entrega
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]+")


def sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto)
                   if not unicodedata.combining(c))


def chave(texto):
    """Chave de comparação: minúscula, sem acento, sem espaços nas pontas."""
    if pd.isna(texto):
        return np.nan
    return sem_acento(str(texto).strip().lower())


def grafia_canonica(serie: pd.Series) -> pd.Series:
    """Para cada nome, usa a grafia mais frequente entre as variações de caixa/espaço."""
    k = serie.map(chave)
    limpa = serie.str.strip()
    mais_comum = (pd.DataFrame({"k": k, "v": limpa}).dropna()
                  .groupby("k")["v"].agg(lambda s: s.value_counts().index[0]))
    return k.map(mais_comum)


def parse_data(valor: str):
    for fmt in ("%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%y"):
        try:
            return pd.to_datetime(valor, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def normalizar_texto(texto):
    if pd.isna(texto):
        return np.nan
    t = EMOJI_RE.sub(" ", str(texto).lower())
    t = sem_acento(t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def limpar(df: pd.DataFrame):
    df = df.copy()
    log = []

    def registrar(etapa, afetadas, detalhe):
        log.append({"etapa": etapa, "linhas_afetadas": int(afetadas), "detalhe": detalhe})

    n0 = len(df)
    registrar("Base original", n0, f"{df.shape[1]} colunas")

    # 1. duplicatas exatas e IDs repetidos
    df = df.drop_duplicates()
    registrar("Duplicatas exatas removidas", n0 - len(df), "mesma avaliação enviada 2x")
    n1 = len(df)
    df = df.drop_duplicates(subset="id_avaliacao", keep="first")
    registrar("IDs repetidos removidos", n1 - len(df), "mantida a primeira ocorrência")

    # 2. textos categóricos
    antes = df["categoria"].copy()
    df["categoria"] = df["categoria"].map(chave).map(MAPA_CATEGORIA)
    registrar("Categoria padronizada", (antes != df["categoria"]).sum(),
              f"{antes.nunique()} grafias -> {df['categoria'].nunique()} categorias")
    assert df["categoria"].notna().all(), "categoria sem mapeamento"

    for col in ("cidade", "restaurante"):
        antes = df[col].copy()
        df[col] = grafia_canonica(df[col])
        mudou = (antes.fillna("§") != df[col].fillna("§")).sum()
        registrar(f"{col.capitalize()} padronizado(a)", mudou,
                  f"{antes.nunique()} valores distintos -> {df[col].nunique()}")

    # 3. datas em 3 formatos diferentes
    df["data_pedido"] = df["data_pedido"].map(parse_data)
    sem_hora = df["data_pedido"].dt.hour.eq(0) & df["data_pedido"].dt.minute.eq(0)
    registrar("Datas convertidas", len(df), f"3 formatos; {sem_hora.sum()} sem horário (dd/mm/aa)")

    # 4. nota: texto -> inteiro; fora de 1–5 vira nulo
    nota = pd.to_numeric(df["nota"], errors="coerce")
    invalida = ~nota.between(1, 5)
    df["nota"] = nota.where(~invalida).astype("Int64")
    registrar("Notas fora da escala 1–5 anuladas", invalida.sum(), "ex.: 0, 6, 10")

    # 5. valor: 'R$ 27,67' -> 27.67
    valor = df["valor_pedido"].astype("string")
    com_rs = valor.str.contains("R\\$", na=False).sum()
    df["valor_pedido"] = pd.to_numeric(
        valor.str.replace("R$", "", regex=False).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce")
    registrar("Valores com 'R$' e vírgula convertidos", com_rs, "texto -> número")

    # 6. tempo de entrega implausível
    t = pd.to_numeric(df["tempo_entrega_min"], errors="coerce")
    fora = t.notna() & ~t.between(TEMPO_MIN, TEMPO_MAX)
    df["tempo_entrega_min"] = t.where(~fora)
    registrar("Tempos de entrega implausíveis anulados", fora.sum(),
              f"fora de {TEMPO_MIN}–{TEMPO_MAX} min (ex.: -5, 0, 999)")

    # 7. comentários vazios disfarçados
    c = df["comentario"].astype("string").str.strip()
    vazio = c.str.lower().isin(COMENTARIOS_VAZIOS)
    df["comentario"] = c.where(~vazio)
    registrar("Comentários sem conteúdo anulados", vazio.sum(), "'', '.', '-', 'ok'")

    # 8. colunas derivadas
    df["tem_comentario"] = df["comentario"].notna()
    df["comentario_norm"] = df["comentario"].map(normalizar_texto)
    df["hora"] = df["data_pedido"].dt.hour.where(~sem_hora)
    df["dia_semana"] = df["data_pedido"].dt.dayofweek
    df["fim_de_semana"] = df["dia_semana"] >= 4      # sexta a domingo
    df["mes"] = df["data_pedido"].dt.to_period("M").astype(str)
    df["sentimento_nota"] = pd.cut(df["nota"].astype(float), [0, 2, 3, 5],
                                   labels=["negativa", "neutra", "positiva"])
    df["faixa_entrega"] = pd.cut(df["tempo_entrega_min"], [0, 30, 40, 50, 60, 200],
                                 labels=["até 30", "31–40", "41–50", "51–60", "60+"])
    registrar("Colunas derivadas criadas", len(df),
              "hora, dia_semana, fim_de_semana, mes, sentimento_nota, faixa_entrega, comentario_norm")

    registrar("Base final", len(df), f"{df.shape[1]} colunas")
    return df.reset_index(drop=True), pd.DataFrame(log)


def main():
    bruto = pd.read_csv(ENTRADA, dtype=str)
    limpo, log = limpar(bruto)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    limpo.to_csv(SAIDA, index=False)
    log.to_csv(RELATORIO, index=False)
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
