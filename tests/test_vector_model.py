"""Testes unitários para o Modelo Vetorial."""

import pytest
import numpy as np
from src.vector_model import VectorSpaceModel
from src.preprocessor import CONFIG_WITH_STOP_WITH_STEM, CONFIG_NO_STOP_NO_STEM
from src.dataset import load_cranfield_dataset


@pytest.fixture
def toy_corpus():
    return {
        "d1": "experimental aerodynamics of high speed wing",
        "d2": "heat conduction in composite slabs and thermal stress",
        "d3": "aeroelastic flutter of supersonic aircraft wings",
    }


def test_vector_model_toy_ranking(toy_corpus):
    model = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM)
    model.fit(toy_corpus)

    results = model.search("aerodynamics of high speed wing", top_k=3)
    assert len(results) == 3
    
    top_did, top_score = results[0]
    assert top_did == "d1"
    assert top_score > 0.5
    
    d2_score = next(score for did, score in results if did == "d2")
    assert d2_score == 0.0


def test_explain_score_mathematical_consistency(toy_corpus):
    model = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM)
    model.fit(toy_corpus)

    explanation = model.explain_score("aerodynamics wing", "d1")
    assert explanation["doc_id"] == "d1"
    assert explanation["cosine_similarity"] > 0
    assert explanation["matching_terms_count"] >= 2
    
    # A soma das contribuições termo a termo deve ser igual à similaridade do cosseno
    sum_contributions = sum(t["contribution"] for t in explanation["terms"])
    assert np.isclose(sum_contributions, explanation["cosine_similarity"], atol=1e-5)


def test_vector_model_cranfield_integration():
    docs, queries, _ = load_cranfield_dataset()
    doc_texts = {did: doc.full_text for did, doc in docs.items()}

    model = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM)
    model.fit(doc_texts)

    query_1_text = queries["1"].text
    results = model.search(query_1_text, top_k=10)

    assert len(results) == 10
    # Verifica monotonicidade decrescente dos scores
    scores = [score for _, score in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    # Pelo menos um documento deve ter score positivo
    assert scores[0] > 0.0


def test_batch_search_equivalence(toy_corpus):
    model = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM)
    model.fit(toy_corpus)

    queries = {
        "q1": "aerodynamics wing",
        "q2": "thermal stress heat",
    }
    
    batch_results = model.batch_search(queries, top_k=3)
    
    for qid, qtext in queries.items():
        single_result = model.search(qtext, top_k=3)
        assert len(batch_results[qid]) == len(single_result)
        for (did_b, sc_b), (did_s, sc_s) in zip(batch_results[qid], single_result):
            assert did_b == did_s
            assert np.isclose(sc_b, sc_s, atol=1e-6)
