"""
tests/test_bm25.py
Inclui verificação analítica manual, verificação de parâmetros k1 e b, e integração com Cranfield.
"""

import math
import numpy as np
import pytest

from src.bm25 import BM25Model
from src.preprocessor import CONFIG_NO_STOP_NO_STEM, CONFIG_WITH_STOP_WITH_STEM
from src.dataset import load_cranfield_dataset


@pytest.fixture
def toy_corpus():
    return {
        "d1": "aerodynamics wing",
        "d2": "wing wing wing flight",
    }


def test_bm25_hand_calculated_math(toy_corpus):
    """
    Verificação analítica com cálculo manual:
    N = 2
    d1: 'aerodynamics wing' -> |d1| = 2
    d2: 'wing wing wing flight' -> |d2| = 4
    avdl = (2 + 4) / 2 = 3.0
    
    k1 = 1.2, b = 0.75
    Para o termo 'aerodynamics':
      df = 1
      IDF = ln( (2 - 1 + 0.5) / (1 + 0.5) + 1 ) = ln( 1.5/1.5 + 1 ) = ln(2.0) = 0.69314718
      f(t, d1) = 1
      B(d1) = 1 - 0.75 + 0.75 * (2.0 / 3.0) = 0.25 + 0.50 = 0.75
      TF_comp = (1 * 2.2) / (1 + 1.2 * 0.75) = 2.2 / 1.90 = 1.157894737
      Score esperado = 0.69314718 * 1.157894737 = 0.80259147
    """
    model = BM25Model(k1=1.2, b=0.75, config=CONFIG_NO_STOP_NO_STEM)
    model.fit(toy_corpus)

    assert model.num_docs == 2
    assert np.isclose(model.avdl, 3.0)
    assert np.isclose(model.idf["aerodynamics"], math.log(2.0))

    results = model.search("aerodynamics", top_k=2)
    top_did, top_score = results[0]

    expected_score = math.log(2.0) * (2.2 / 1.9)
    assert top_did == "d1"
    assert np.isclose(top_score, expected_score, atol=1e-5)
    assert results[1][1] == 0.0


def test_bm25_length_normalization_effect(toy_corpus):
    """
    Para a query 'wing':
    - d1 tem 1 'wing' e tamanho 2 (curto).
    - d2 tem 3 'wing' e tamanho 4 (longo).
    
    Quando b = 0 (sem penalização por comprimento):
    d2 deve ter score bem maior que d1 porque tem 3 ocorrências contra 1.
    
    Quando b = 1.0 (penalização máxima por comprimento):
    O comprimento de d2 (4 vs avdl 3) reduz a vantagem das ocorrências repetidas.
    """
    model_b0 = BM25Model(k1=1.2, b=0.0, config=CONFIG_NO_STOP_NO_STEM)
    model_b0.fit(toy_corpus)
    res_b0 = dict(model_b0.search("wing"))

    model_b1 = BM25Model(k1=1.2, b=1.0, config=CONFIG_NO_STOP_NO_STEM)
    model_b1.fit(toy_corpus)
    res_b1 = dict(model_b1.search("wing"))

    # A razão score(d2)/score(d1) com b=0 deve ser significativamente maior do que com b=1
    ratio_b0 = res_b0["d2"] / res_b0["d1"]
    ratio_b1 = res_b1["d2"] / res_b1["d1"]
    assert ratio_b0 > ratio_b1, "A penalização por comprimento b=1 deveria reduzir o score relativo de d2"


def test_bm25_explain_score_consistency(toy_corpus):
    model = BM25Model(k1=1.2, b=0.75, config=CONFIG_NO_STOP_NO_STEM)
    model.fit(toy_corpus)

    explanation = model.explain_score("aerodynamics wing", "d1")
    search_score = dict(model.search("aerodynamics wing"))["d1"]

    assert explanation["doc_id"] == "d1"
    assert np.isclose(explanation["bm25_score"], search_score, atol=1e-5)
    assert explanation["matching_terms_count"] == 2
    sum_terms = sum(t["contribution"] for t in explanation["terms"])
    assert np.isclose(sum_terms, search_score, atol=1e-5)


def test_bm25_cranfield_integration():
    docs, queries, _ = load_cranfield_dataset()
    doc_texts = {did: doc.full_text for did, doc in docs.items()}

    model = BM25Model(k1=1.2, b=0.75, config=CONFIG_WITH_STOP_WITH_STEM)
    model.fit(doc_texts)

    query_1_text = queries["1"].text
    results = model.search(query_1_text, top_k=10)

    assert len(results) == 10
    scores = [s for _, s in results]
    # Monotonicidade decrescente
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    assert scores[0] > 0.0
