"""Testes unitários para o módulo de métricas de avaliação."""

import math
import numpy as np
import pytest

from src.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    reciprocal_rank,
    ndcg_at_k,
    evaluate_system,
)


@pytest.fixture
def sample_qrels():
    # 4 documentos relevantes com diferentes graus de relevância Cleverdon
    return {
        "d1": 1,   # Resposta completa (ganho 4)
        "d2": 2,   # Alta relevância (ganho 3)
        "d3": 3,   # Útil (ganho 2)
        "d4": 4,   # Mínimo interesse (ganho 1)
        "d_neg": -1  # Explicitamente não relevante
    }


def test_hand_calculated_metrics(sample_qrels):
    # Ranking com relevantes em posições 1 e 3
    ranking = ["d1", "x1", "d2", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]

    # 1. P@5: 2 relevantes em 5 docs = 2/5 = 0.4
    p5 = precision_at_k(ranking, sample_qrels, k=5)
    assert np.isclose(p5, 0.4)

    # 2. R@5: 2 relevantes encontrados de 4 existentes = 2/4 = 0.5
    r5 = recall_at_k(ranking, sample_qrels, k=5)
    assert np.isclose(r5, 0.5)

    # 3. F1@5: 2 * (0.4 * 0.5) / (0.4 + 0.5) = 0.4 / 0.9 = 4/9
    f1_5 = f1_at_k(ranking, sample_qrels, k=5)
    assert np.isclose(f1_5, 4.0 / 9.0)

    # 4. Average Precision (AP):
    # Relevantes nos ranks 1 e 3:
    # P@1 = 1/1 = 1.0
    # P@3 = 2/3
    # AP = (1.0 + 2/3) / 4 = (5/3) / 4 = 5/12
    ap = average_precision(ranking, sample_qrels)
    assert np.isclose(ap, 5.0 / 12.0)

    # 5. Reciprocal Rank (RR): primeiro relevante no rank 1
    rr = reciprocal_rank(ranking, sample_qrels)
    assert np.isclose(rr, 1.0)


def test_ndcg_graded_math(sample_qrels):
    ranking = ["d1", "x1", "d2"]
    # d1 ganho 4: (2^4 - 1)/log2(2) = 15 / 1 = 15.0
    # x1 ganho 0: 0
    # d2 ganho 3: (2^3 - 1)/log2(4) = 7 / 2 = 3.5
    # DCG@3 = 18.5
    # Ideal: d1 (ganho 4) no rank 1, d2 (ganho 3) no rank 2, d3 (ganho 2) no rank 3
    # IDCG@3 = 15/1 + 7/log2(3) + 3/log2(4) = 16.5 + 7/log2(3)
    idcg_expected = 15.0 + (7.0 / math.log2(3)) + (3.0 / 2.0)
    dcg_expected = 15.0 + 0.0 + 3.5
    expected_ndcg = dcg_expected / idcg_expected

    ndcg_res = ndcg_at_k(ranking, sample_qrels, k=3, graded=True)
    assert np.isclose(ndcg_res, expected_ndcg, atol=1e-5)


def test_edge_cases():
    empty_qrels = {}
    ranking = ["d1", "d2"]
    
    assert precision_at_k(ranking, empty_qrels, k=10) == 0.0
    assert recall_at_k(ranking, empty_qrels, k=10) == 0.0
    assert average_precision(ranking, empty_qrels) == 0.0
    assert ndcg_at_k(ranking, empty_qrels, k=10) == 0.0
    assert reciprocal_rank(ranking, empty_qrels) == 0.0

    # Ranking perfeito (todos os relevantes nas primeiras posições)
    perfect_qrels = {"d1": 1, "d2": 1}
    perfect_ranking = ["d1", "d2", "d3"]
    assert average_precision(perfect_ranking, perfect_qrels) == 1.0
    assert ndcg_at_k(perfect_ranking, perfect_qrels, k=2, graded=False) == 1.0


def test_evaluate_system_aggregation(sample_qrels):
    rankings = {
        "1": [("d1", 0.9), ("x1", 0.8), ("d2", 0.7)],
        "2": [("x1", 0.9), ("x2", 0.8)],  # zero relevantes
    }
    qrels_dict = {
        "1": sample_qrels,
        "2": {"d9": 1},
    }

    df, agg = evaluate_system(rankings, qrels_dict, k=3)
    assert len(df) == 2
    assert "query_id" in df.columns
    assert "MAP" in agg
    assert "Precision@3" in agg
    assert "Recall@3" in agg
    assert "NDCG@3" in agg
    assert "MRR" in agg
    assert 0.0 <= agg["MAP"] <= 1.0
