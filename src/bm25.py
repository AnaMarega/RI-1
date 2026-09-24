"""Modelo Probabilístico BM25 (Robertson & Zaragoza, 2009)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.preprocessor import PreprocessingConfig, TextPreprocessor, CONFIG_WITH_STOP_WITH_STEM


class BM25Model:
    """
    Modelo de Recuperação de Informação Okapi BM25.

    Score:
        BM25(d, q) = sum_{t in q} IDF(t) * [ (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * B(d)) ]
        onde B(d) = 1 - b + b * (|d| / avdl)

    IDF (Robertson-Spärck Jones):
        IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        config: PreprocessingConfig = CONFIG_WITH_STOP_WITH_STEM,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.config = config
        self.preprocessor = TextPreprocessor(config)

        self.doc_ids: List[str] = []
        self.doc_index_map: Dict[str, int] = {}
        self.doc_lengths: np.ndarray = np.array([])
        self.avdl: float = 0.0
        self.num_docs: int = 0

        # term -> [(doc_index, term_frequency), ...]
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted: bool = False

    def fit(self, documents: Dict[str, str]) -> "BM25Model":
        """
        Indexa a coleção de documentos e calcula estatísticas globais (avdl, df, idf).
        
        Args:
            documents: Mapeamento {doc_id: texto_do_documento}
        """
        self.doc_ids = list(documents.keys())
        self.num_docs = len(self.doc_ids)
        self.doc_index_map = {did: idx for idx, did in enumerate(self.doc_ids)}

        doc_lengths_list: List[int] = []
        self.inverted_index = {}
        self.doc_term_freqs = []
        self.df = {}

        for idx, did in enumerate(self.doc_ids):
            text = documents[did]
            tokens = self.preprocessor.process(text)
            doc_len = len(tokens)
            doc_lengths_list.append(doc_len)

            counts = Counter(tokens)
            self.doc_term_freqs.append(counts)

            for term, freq in counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    self.df[term] = 0
                self.inverted_index[term].append((idx, freq))
                self.df[term] += 1

        self.doc_lengths = np.array(doc_lengths_list, dtype=np.float64)
        self.avdl = float(np.mean(self.doc_lengths)) if self.num_docs > 0 else 0.0

        self.idf = {}
        for term, doc_freq in self.df.items():
            val = (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5)
            self.idf[term] = math.log(val + 1.0)

        self._fitted = True
        return self

    def set_parameters(self, k1: Optional[float] = None, b: Optional[float] = None) -> None:
        """Atualiza os parâmetros k1 e b sem reindexar a coleção."""
        if k1 is not None:
            self.k1 = float(k1)
        if b is not None:
            self.b = float(b)

    def search(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Calcula os scores BM25 para uma consulta e retorna os documentos ranqueados.
        """
        if not self._fitted:
            raise RuntimeError("O modelo precisa ser indexado com fit() antes de realizar buscas.")

        query_tokens = self.preprocessor.process(query_text)
        if not query_tokens:
            return [(did, 0.0) for did in (self.doc_ids[:top_k] if top_k else self.doc_ids)]

        q_counts = Counter(query_tokens)
        scores = np.zeros(self.num_docs, dtype=np.float64)

        for term, q_tf in q_counts.items():
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            doc_indices = np.array([p[0] for p in postings], dtype=np.int64)
            term_freqs = np.array([p[1] for p in postings], dtype=np.float64)
            lengths = self.doc_lengths[doc_indices]

            if self.avdl > 0:
                length_norm = 1.0 - self.b + self.b * (lengths / self.avdl)
            else:
                length_norm = 1.0

            tf_component = (term_freqs * (self.k1 + 1.0)) / (term_freqs + self.k1 * length_norm)
            term_scores = term_idf * tf_component * q_tf
            scores[doc_indices] += term_scores

        ranked_indices = np.argsort(-scores)
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

    def batch_search(
        self, queries: Dict[str, str], top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Executa a busca para todas as consultas informadas.
        """
        return {qid: self.search(qtext, top_k=top_k) for qid, qtext in queries.items()}

    def explain_score(self, query_text: str, doc_id: str) -> Dict[str, Any]:
        """Retorna a decomposição termo a termo do score BM25 para um documento."""
        if not self._fitted:
            raise RuntimeError("Modelo não indexado.")

        if doc_id not in self.doc_index_map:
            raise ValueError(f"Documento {doc_id} não encontrado.")

        doc_idx = self.doc_index_map[doc_id]
        doc_len = float(self.doc_lengths[doc_idx])
        doc_counts = self.doc_term_freqs[doc_idx]

        query_tokens = self.preprocessor.process(query_text)
        q_counts = Counter(query_tokens)

        length_factor = 1.0 - self.b + self.b * (doc_len / self.avdl) if self.avdl > 0 else 1.0

        term_breakdown: List[Dict[str, Any]] = []
        total_score = 0.0

        for term, q_tf in q_counts.items():
            tf_doc = doc_counts.get(term, 0)
            term_idf = self.idf.get(term, 0.0)

            if tf_doc > 0:
                tf_sat = (tf_doc * (self.k1 + 1.0)) / (tf_doc + self.k1 * length_factor)
                contrib = float(term_idf * tf_sat * q_tf)
            else:
                tf_sat = 0.0
                contrib = 0.0

            total_score += contrib
            term_breakdown.append({
                "term": term,
                "in_document": bool(tf_doc > 0),
                "query_tf": q_tf,
                "doc_tf": tf_doc,
                "idf": term_idf,
                "length_factor_B": float(length_factor),
                "tf_saturation_component": float(tf_sat),
                "contribution": contrib,
            })

        term_breakdown.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "query": query_text,
            "doc_id": doc_id,
            "k1": self.k1,
            "b": self.b,
            "doc_length": int(doc_len),
            "avdl": float(self.avdl),
            "length_factor_B": float(length_factor),
            "bm25_score": float(total_score),
            "matching_terms_count": sum(1 for t in term_breakdown if t["in_document"]),
            "terms": term_breakdown,
        }
