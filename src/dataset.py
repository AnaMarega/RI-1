"""
src/dataset.py
Módulo para download, carregamento e parsing da coleção Cranfield.
Permite carregar documentos (1400), consultas (225) e julgamentos de relevância (qrels).
"""

from __future__ import annotations

import os
import re
import urllib.request
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Document:
    """Representação de um documento da coleção Cranfield."""
    doc_id: str
    title: str = ""
    author: str = ""
    bib: str = ""
    text: str = ""
    
    @property
    def full_text(self) -> str:
        """Combina título e corpo do documento."""
        parts = [p.strip() for p in (self.title, self.text) if p.strip()]
        return " ".join(parts)


@dataclass
class Query:
    """Representação de uma consulta (query) da coleção Cranfield."""
    query_id: str          # ID sequencial correspondente ao cranqrel (1 a 225)
    text: str
    raw_id: str = ""       # ID original no arquivo cran.qry (ex: '001', '004')


# URLs canônicas da coleção Cranfield
CRANFIELD_TAR_URL = "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
CRANFIELD_FALLBACK_URLS = {
    "cran.all.1400": "https://raw.githubusercontent.com/arosh/BM25-latency-benchmark/master/data/cran.all.1400",
    "cran.qry": "https://raw.githubusercontent.com/arosh/BM25-latency-benchmark/master/data/cran.qry",
    "cranqrel": "https://raw.githubusercontent.com/arosh/BM25-latency-benchmark/master/data/cranqrel",
}


def download_cranfield(target_dir: Path) -> None:
    """
    Garante a presença dos arquivos originais da coleção Cranfield em target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    required_files = ["cran.all.1400", "cran.qry", "cranqrel"]
    
    if all((target_dir / fname).exists() for fname in required_files):
        return

    print(f"[Dataset] Baixando coleção Cranfield para {target_dir}...")
    tar_path = target_dir / "cran.tar.gz"
    try:
        urllib.request.urlretrieve(CRANFIELD_TAR_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        if tar_path.exists():
            tar_path.unlink()
        print("[Dataset] Coleção Cranfield descompactada com sucesso.")
    except Exception as e:
        print(f"[Dataset] Aviso: Falha ao baixar de Glasgow ({e}). Tentando espelho alternativo...")
        for fname, url in CRANFIELD_FALLBACK_URLS.items():
            dest = target_dir / fname
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
                print(f"[Dataset] Baixado: {fname}")


def parse_cranfield_docs(filepath: Path) -> Dict[str, Document]:
    """
    Realiza o parsing de cran.all.1400 com as marcações .I, .T, .A, .B, .W.
    Retorna dicionário mapeando doc_id -> Document.
    """
    docs: Dict[str, Document] = {}
    current_doc: Optional[Document] = None
    current_field: Optional[str] = None
    content_buffer: List[str] = []

    def flush_field():
        if current_doc is None or current_field is None:
            return
        text = " ".join(" ".join(content_buffer).split())
        if current_field == ".T":
            current_doc.title = text
        elif current_field == ".A":
            current_doc.author = text
        elif current_field == ".B":
            current_doc.bib = text
        elif current_field == ".W":
            current_doc.text = text

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(".I"):
                flush_field()
                if current_doc is not None:
                    docs[current_doc.doc_id] = current_doc
                
                parts = stripped.split()
                raw_num = parts[1] if len(parts) > 1 else str(len(docs) + 1)
                doc_id = str(int(raw_num))
                current_doc = Document(doc_id=doc_id)
                current_field = None
                content_buffer = []
            elif stripped in {".T", ".A", ".B", ".W"}:
                flush_field()
                current_field = stripped
                content_buffer = []
            else:
                content_buffer.append(line.rstrip("\n"))

        flush_field()
        if current_doc is not None:
            docs[current_doc.doc_id] = current_doc

    return docs


def parse_cranfield_queries(filepath: Path) -> Dict[str, Query]:
    """
    Realiza o parsing de cran.qry com marcações .I e .W.
    Mapeia query_id sequencialmente (1 a 225) conforme cranqrel e armazena raw_id.
    """
    queries: Dict[str, Query] = {}
    current_query: Optional[Query] = None
    content_buffer: List[str] = []
    seq_counter = 0

    def flush_query():
        if current_query is not None:
            clean_text = " ".join(" ".join(content_buffer).split())
            current_query.text = clean_text
            queries[current_query.query_id] = current_query

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(".I"):
                flush_query()
                seq_counter += 1
                parts = stripped.split()
                raw_id = parts[1] if len(parts) > 1 else str(seq_counter)
                query_id = str(seq_counter)
                current_query = Query(query_id=query_id, text="", raw_id=raw_id)
                content_buffer = []
            elif stripped == ".W":
                content_buffer = []
            else:
                content_buffer.append(line.rstrip("\n"))

        flush_query()

    return queries


def parse_cranfield_qrels(filepath: Path) -> Dict[str, Dict[str, int]]:
    """
    Realiza o parsing de cranqrel.
    Retorna dicionário mapeando query_id -> {doc_id: relevance_score}.
    
    - Relevante: score >= 1 (escala 1 a 4)
    - Julgamentos com valor -1 e não julgados são considerados não relevantes.
    """
    qrels: Dict[str, Dict[str, int]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if len(parts) == 3:
                qid, did, rel_str = parts[0], parts[1], parts[2]
            elif len(parts) >= 4:
                if parts[1] == "0":
                    qid, did, rel_str = parts[0], parts[2], parts[3]
                else:
                    qid, did, rel_str = parts[0], parts[1], parts[3]
            else:
                continue

            try:
                qid_norm = str(int(qid))
                did_norm = str(int(did))
                rel = int(rel_str)
            except ValueError:
                continue

            if qid_norm not in qrels:
                qrels[qid_norm] = {}
            qrels[qid_norm][did_norm] = rel

    return qrels


def load_cranfield_dataset(data_dir: Optional[Path] = None) -> Tuple[Dict[str, Document], Dict[str, Query], Dict[str, Dict[str, int]]]:
    """Garante o download e carrega documentos, consultas e qrels."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    download_cranfield(data_dir)
    
    docs = parse_cranfield_docs(data_dir / "cran.all.1400")
    queries = parse_cranfield_queries(data_dir / "cran.qry")
    qrels = parse_cranfield_qrels(data_dir / "cranqrel")

    return docs, queries, qrels


if __name__ == "__main__":
    docs, queries, qrels = load_cranfield_dataset()
    print(f"Total de documentos carregados: {len(docs)}")
    print(f"Total de consultas carregadas: {len(queries)}")
    print(f"Total de consultas com julgamentos (qrels): {len(qrels)}")
    
    # Validação do alinhamento
    print("\n--- Verificação de Alinhamento das 5 Primeiras Consultas ---")
    for qid in ["1", "2", "3", "4", "5"]:
        q = queries[qid]
        q_qrels = qrels.get(qid, {})
        rel_docs = [did for did, score in q_qrels.items() if score >= 1]
        print(f"Consulta {qid} (original .I {q.raw_id}): '{q.text[:55]}...' -> {len(rel_docs)} docs relevantes")
        if rel_docs:
            sample_did = rel_docs[0]
            print(f"   Exemplo de Doc Relevante ({sample_did}): '{docs[sample_did].title[:60]}...'")
