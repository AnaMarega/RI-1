"""Testes para o módulo src/dataset.py."""

from pathlib import Path
import pytest
from src.dataset import load_cranfield_dataset, Document, Query


@pytest.fixture(scope="module")
def cranfield_data():
    docs, queries, qrels = load_cranfield_dataset()
    return docs, queries, qrels


def test_dataset_counts(cranfield_data):
    docs, queries, qrels = cranfield_data
    assert len(docs) == 1400, f"Esperado 1400 documentos, encontrado {len(docs)}"
    assert len(queries) == 225, f"Esperado 225 consultas, encontrado {len(queries)}"
    assert len(qrels) == 225, f"Esperado 225 conjuntos de qrels, encontrado {len(qrels)}"


def test_document_integrity(cranfield_data):
    docs, _, _ = cranfield_data
    doc_1 = docs["1"]
    assert isinstance(doc_1, Document)
    assert doc_1.doc_id == "1"
    assert "experimental investigation" in doc_1.title.lower()
    assert len(doc_1.full_text) > len(doc_1.title)


def test_query_qrel_alignment(cranfield_data):
    docs, queries, qrels = cranfield_data
    for i in range(1, 226):
        qid = str(i)
        assert qid in queries, f"Consulta {qid} não encontrada em queries"
        assert qid in qrels, f"Consulta {qid} não encontrada em qrels"
        assert len(qrels[qid]) > 0, f"Consulta {qid} tem 0 documentos avaliados"
        for did, score in qrels[qid].items():
            assert did in docs, f"Documento {did} referenciado na query {qid} não existe"
            assert score in {-1, 1, 2, 3, 4}, f"Score {score} inválido para query {qid}, doc {did}"
