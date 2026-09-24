"""Módulo de métricas de avaliação (Precision@k, Recall@k, F1@k, MAP, NDCG@k, MRR)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def get_relevant_doc_ids(qrels_for_query: Dict[str, int]) -> set[str]:
    """
    Retorna o conjunto de IDs de documentos relevantes para uma consulta.
    Escala Cleverdon:
    - Relevante: grau >= 1 (1 a 4)
    - Não relevante: grau == -1 e documentos não avaliados
    """
    return {did for did, score in qrels_for_query.items() if score >= 1}


def precision_at_k(
    ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
) -> float:
    """
    Precision@k: Fração dos top-k documentos retornados que são relevantes.
    P@k = |Relevantes no Top-k| / k
    """
    if k <= 0:
        return 0.0
    relevant_ids = get_relevant_doc_ids(qrels)
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    
    hits = sum(1 for did in top_k if did in relevant_ids)
    return hits / k


def recall_at_k(
    ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
) -> float:
    """
    Recall@k: Fração de todos os documentos relevantes da coleção recuperados no Top-k.
    R@k = |Relevantes no Top-k| / |Total de Relevantes|
    """
    relevant_ids = get_relevant_doc_ids(qrels)
    total_relevant = len(relevant_ids)
    if total_relevant == 0:
        return 0.0

    top_k = ranked_doc_ids[:k]
    hits = sum(1 for did in top_k if did in relevant_ids)
    return hits / total_relevant


def f1_at_k(
    ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
) -> float:
    """
    F1-Score@k: Média harmônica entre Precision@k e Recall@k.
    """
    p = precision_at_k(ranked_doc_ids, qrels, k=k)
    r = recall_at_k(ranked_doc_ids, qrels, k=k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def average_precision(
    ranked_doc_ids: List[str], qrels: Dict[str, int]
) -> float:
    """
    Average Precision (AP) para uma única consulta:
    AP = (1 / |Total_Rel|) * sum_{k=1}^N (P@k * rel(k))
    onde rel(k) é 1 se o documento na posição k é relevante, 0 caso contrário.
    """
    relevant_ids = get_relevant_doc_ids(qrels)
    total_relevant = len(relevant_ids)
    if total_relevant == 0:
        return 0.0

    cumulative_hits = 0
    precision_sum = 0.0

    for rank_idx, did in enumerate(ranked_doc_ids, start=1):
        if did in relevant_ids:
            cumulative_hits += 1
            precision_at_rank = cumulative_hits / rank_idx
            precision_sum += precision_at_rank

    return precision_sum / total_relevant


def reciprocal_rank(
    ranked_doc_ids: List[str], qrels: Dict[str, int]
) -> float:
    """
    Reciprocal Rank (RR): Inverso do rank do primeiro documento relevante retornado.
    RR = 1 / rank_primeiro_relevante (0.0 se nenhum relevante for retornado).
    """
    relevant_ids = get_relevant_doc_ids(qrels)
    for rank_idx, did in enumerate(ranked_doc_ids, start=1):
        if did in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(
    ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10, graded: bool = True
) -> float:
    """
    Normalized Discounted Cumulative Gain no corte k (NDCG@k).
    
    Na escala Cleverdon:
    1 = Resposta completa (mais relevante)
    2 = Alta relevância
    3 = Útil / contexto
    4 = Mínimo interesse
    -1 ou não julgado = Não relevante (ganho 0)
    
    Para relevância graduada:
    ganho(1) = 4, ganho(2) = 3, ganho(3) = 2, ganho(4) = 1, outros = 0.
    
    Fórmula:
    DCG@k = sum_{i=1}^k (2^ganho(i) - 1) / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k
    """
    if k <= 0:
        return 0.0

    def get_gain(did: str) -> float:
        raw_score = qrels.get(did, 0)
        if raw_score <= 0:
            return 0.0
        if graded:
            # Inverte a escala Cleverdon (1 é melhor -> ganho 4)
            # 1 -> 4, 2 -> 3, 3 -> 2, 4 -> 1
            return float(5 - raw_score) if 1 <= raw_score <= 4 else 0.0
        else:
            return 1.0 if raw_score >= 1 else 0.0

    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for i, did in enumerate(top_k, start=1):
        gain = get_gain(did)
        discount = math.log2(i + 1)
        dcg += (2.0 ** gain - 1.0) / discount

    # Ganhos ideais de todos os documentos julgados para esta consulta
    all_gains = [get_gain(did) for did in qrels.keys()]
    all_gains.sort(reverse=True)
    ideal_top_k_gains = all_gains[:k]

    idcg = 0.0
    for i, gain in enumerate(ideal_top_k_gains, start=1):
        discount = math.log2(i + 1)
        idcg += (2.0 ** gain - 1.0) / discount

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_system(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Avalia os rankings produzidos para todas as consultas.
    
    Args:
        rankings: Mapeamento {query_id: [(doc_id, score), ...]}
        qrels: Mapeamento {query_id: {doc_id: score}}
        k: Ponto de corte para métricas @k (padrão 10)
        
    Retorna:
        - per_query_df: DataFrame com métricas individuais por consulta (P@k, R@k, F1@k, AP, NDCG@k, RR)
        - aggregate_metrics: Dicionário com as médias globais (MAP, Mean P@k, Mean R@k, Mean NDCG@k, MRR)
    """
    rows: List[Dict[str, Any]] = []

    for qid in sorted(rankings.keys(), key=lambda x: int(x)):
        ranked_list = [did for did, _ in rankings[qid]]
        q_qrels = qrels.get(qid, {})

        p_k = precision_at_k(ranked_list, q_qrels, k=k)
        r_k = recall_at_k(ranked_list, q_qrels, k=k)
        f1_k = f1_at_k(ranked_list, q_qrels, k=k)
        ap = average_precision(ranked_list, q_qrels)
        ndcg_k = ndcg_at_k(ranked_list, q_qrels, k=k, graded=True)
        rr = reciprocal_rank(ranked_list, q_qrels)
        total_rel = len(get_relevant_doc_ids(q_qrels))

        rows.append({
            "query_id": qid,
            "total_relevant": total_rel,
            f"P@{k}": p_k,
            f"R@{k}": r_k,
            f"F1@{k}": f1_k,
            "AP": ap,
            f"NDCG@{k}": ndcg_k,
            "RR": rr,
        })

    df = pd.DataFrame(rows)

    aggregate = {
        f"Precision@{k}": float(df[f"P@{k}"].mean()),
        f"Recall@{k}": float(df[f"R@{k}"].mean()),
        f"F1@{k}": float(df[f"F1@{k}"].mean()),
        "MAP": float(df["AP"].mean()),
        f"NDCG@{k}": float(df[f"NDCG@{k}"].mean()),
        "MRR": float(df["RR"].mean()),
    }

    return df, aggregate
