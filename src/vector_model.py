"""
src/vector_model.py
Implementação do Modelo Vetorial (Vector Space Model - VSM) com ponderação TF-IDF
e similaridade do cosseno para recuperação e ranqueamento de documentos.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessor import PreprocessingConfig, TextPreprocessor, CONFIG_WITH_STOP_WITH_STEM


class VectorSpaceModel:
    """
    Modelo Vetorial de Recuperação de Informação baseado em TF-IDF e Similaridade do Cosseno.

    Conceitos matemáticos:
    1. Representação do Documento e da Consulta:
       Cada documento d e consulta q é representado como um vetor esparso de pesos TF-IDF
       no espaço vetorial V (vocabulário):
       v_d = [w_{t1, d}, w_{t2, d}, ..., w_{tV, d}]
       v_q = [w_{t1, q}, w_{t2, q}, ..., w_{tV, q}]

    2. Ponderação TF-IDF:
       tf(t, d): frequência do termo t no documento d
       idf(t): frequência inversa nos documentos = log((1 + N) / (1 + df(t))) + 1
       peso bruto = tf(t, d) * idf(t)

    3. Normalização Euclidiana (L2):
       v_norm = v / ||v||_2, onde ||v||_2 = sqrt(sum(w_i^2))
       A normalização elimina o viés de tamanho de documentos no cálculo de similaridade angular.

    4. Similaridade do Cosseno:
       Com vetores normalizados em L2, o cosseno reduz-se ao produto escalar:
       Cosine(d, q) = sum_{t in d cap q} (v_norm_{t, d} * v_norm_{t, q})
    """

    def __init__(self, config: PreprocessingConfig = CONFIG_WITH_STOP_WITH_STEM) -> None:
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        
        # O vetorizador utiliza o preprocessor customizado para garantir que os tokens
        # de documentos e consultas passem pelas mesmas transformações
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.process,
            lowercase=False,         # Já tratado no preprocessor
            token_pattern=None,      # Usa a tokenização do preprocessor
            norm="l2",               # Normalização L2 para similaridade do cosseno
            smooth_idf=True,
            sublinear_tf=False
        )
        self.doc_ids: List[str] = []
        self.doc_matrix: Optional[Any] = None
        self._fitted: bool = False

    def fit(self, documents: Dict[str, str]) -> "VectorSpaceModel":
        """
        Indexa a coleção de documentos, calculando vocabulário e a matriz de pesos TF-IDF.
        
        Args:
            documents: Mapeamento {doc_id: texto_do_documento}
        """
        self.doc_ids = list(documents.keys())
        corpus = [documents[did] for did in self.doc_ids]
        
        # Matriz esparsa (N_docs x N_termos) com vetores normalizados em L2
        self.doc_matrix = self.vectorizer.fit_transform(corpus)
        self._fitted = True
        return self

    def search(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Ranqueia os documentos para uma dada consulta com base na similaridade do cosseno.
        
        Retorna:
            Lista ordenada de tuplas (doc_id, score_cosseno) decrescente pelo score.
        """
        if not self._fitted or self.doc_matrix is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de realizar buscas.")

        # Vetoriza a consulta no mesmo espaço dimensional
        query_vec = self.vectorizer.transform([query_text])
        
        # Caso a consulta não tenha termos presentes no vocabulário
        if query_vec.nnz == 0:
            return [(did, 0.0) for did in (self.doc_ids[:top_k] if top_k else self.doc_ids)]

        # Produto escalar entre matriz de docs normalizados e vetor de consulta normalizado
        # Equivale a cosine_similarity(self.doc_matrix, query_vec)
        scores = (self.doc_matrix * query_vec.T).toarray().flatten()

        # Ordenação decrescente de scores
        ranked_indices = np.argsort(-scores)

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

    def batch_search(
        self, queries: Dict[str, str], top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Executa buscas em lote para múltiplas consultas de forma vetorizada eficiente.
        
        Retorna:
            Mapeamento {query_id: [(doc_id, score), ...]}
        """
        if not self._fitted or self.doc_matrix is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de realizar buscas.")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        # Matriz esparsa de consultas (N_queries x N_termos)
        query_matrices = self.vectorizer.transform(query_texts)
        
        # Multiplicação matricial (N_queries x N_docs)
        similarity_matrix = (query_matrices * self.doc_matrix.T).toarray()
        
        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, qid in enumerate(query_ids):
            scores = similarity_matrix[i]
            ranked_indices = np.argsort(-scores)
            if top_k is not None:
                ranked_indices = ranked_indices[:top_k]
            results[qid] = [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]
            
        return results

    def explain_score(
        self, query_text: str, doc_id: str
    ) -> Dict[str, Any]:
        """
        Explica detalhadamente como o score de similaridade do cosseno foi composto
        termo a termo entre a consulta e o documento especificado.
        Permite auditar a contribuição de cada termo no ranking final.
        """
        if not self._fitted or self.doc_matrix is None:
            raise RuntimeError("Modelo não treinado.")
        
        if doc_id not in self.doc_ids:
            raise ValueError(f"Documento {doc_id} não encontrado na base indexada.")

        doc_idx = self.doc_ids.index(doc_id)
        query_vec = self.vectorizer.transform([query_text])
        doc_vec = self.doc_matrix[doc_idx]
        
        feature_names = self.vectorizer.get_feature_names_out()
        idf_weights = self.vectorizer.idf_
        
        # Termos presentes na consulta
        q_nonzero = query_vec.nonzero()[1]
        
        term_contributions: List[Dict[str, Any]] = []
        total_cosine = 0.0

        for col_idx in q_nonzero:
            term = feature_names[col_idx]
            q_weight = query_vec[0, col_idx]
            d_weight = doc_vec[0, col_idx]
            product = float(q_weight * d_weight)
            total_cosine += product
            
            term_contributions.append({
                "term": term,
                "idf": float(idf_weights[col_idx]),
                "query_weight_l2": float(q_weight),
                "doc_weight_l2": float(d_weight),
                "contribution": product,
                "in_document": bool(d_weight > 0)
            })

        # Ordena termos por maior contribuição para o score
        term_contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "query": query_text,
            "doc_id": doc_id,
            "cosine_similarity": float(total_cosine),
            "matching_terms_count": sum(1 for t in term_contributions if t["in_document"]),
            "terms": term_contributions
        }
