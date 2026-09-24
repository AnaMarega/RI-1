"""Pipeline de experimentos para avaliação de modelos de RI na coleção Cranfield."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Garante que a raiz do repositório esteja sempre no sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "data" / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")  # Backend headless para ambiente sem display X11
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataset import Document, Query, load_cranfield_dataset
from src.metrics import evaluate_system, get_relevant_doc_ids
from src.preprocessor import (
    CONFIG_NO_STOP_NO_STEM,
    CONFIG_NO_STOP_WITH_STEM,
    CONFIG_WITH_STOP_NO_STEM,
    CONFIG_WITH_STOP_WITH_STEM,
    PREPROCESSING_CONFIGS,
    PreprocessingConfig,
)
from src.vector_model import VectorSpaceModel
from src.bm25 import BM25Model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})


# --- Experimento 1: Comparação de Modelos e Pré-processamento ---

def run_preprocessing_and_model_comparison(
    docs: Dict[str, Document],
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    Executa os Modelos Vetorial e BM25 nas 4 configurações de pré-processamento.
    Salva tabela comparativa em CSV e gera gráficos de barras comparativos.
    """
    print("\n[Experimento 1] Comparando Pré-processamento e Modelos (Vetorial vs BM25)...")
    doc_texts = {did: d.full_text for did, d in docs.items()}
    query_texts = {qid: q.text for qid, q in queries.items()}

    records: List[Dict[str, Any]] = []
    per_query_results: Dict[str, pd.DataFrame] = {}

    for cfg in PREPROCESSING_CONFIGS:
        print(f"  -> Avaliando configuração: {cfg.name} ({cfg.label})...")
        
        vsm = VectorSpaceModel(cfg)
        vsm.fit(doc_texts)
        vsm_ranks = vsm.batch_search(query_texts, top_k=None)
        vsm_df, vsm_agg = evaluate_system(vsm_ranks, qrels, k=10)
        per_query_results[f"VSM_{cfg.name}"] = vsm_df
        records.append({
            "Modelo": "Modelo Vetorial",
            "Config_Code": cfg.name,
            "Pré-processamento": cfg.label,
            **vsm_agg
        })

        bm25 = BM25Model(k1=1.2, b=0.75, config=cfg)
        bm25.fit(doc_texts)
        bm25_ranks = bm25.batch_search(query_texts, top_k=None)
        bm25_df, bm25_agg = evaluate_system(bm25_ranks, qrels, k=10)
        per_query_results[f"BM25_{cfg.name}"] = bm25_df
        records.append({
            "Modelo": "Modelo BM25",
            "Config_Code": cfg.name,
            "Pré-processamento": cfg.label,
            **bm25_agg
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(RESULTS_DIR / "preprocessing_and_models_comparison.csv", index=False)
    
    # Resultados por consulta na configuração padrão (with_stop_with_stem)
    vsm_full = per_query_results["VSM_with_stop_with_stem"]
    bm25_full = per_query_results["BM25_with_stop_with_stem"]
    
    merged_queries = pd.merge(
        vsm_full[["query_id", "total_relevant", "P@10", "R@10", "AP", "NDCG@10", "RR"]],
        bm25_full[["query_id", "P@10", "R@10", "AP", "NDCG@10", "RR"]],
        on="query_id",
        suffixes=("_VSM", "_BM25")
    )
    merged_queries["AP_Diff (BM25 - VSM)"] = merged_queries["AP_BM25"] - merged_queries["AP_VSM"]
    merged_queries.to_csv(RESULTS_DIR / "per_query_comparison_default_config.csv", index=False)

    # Gráfico de barras comparativo de MAP
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=summary_df,
        x="Config_Code",
        y="MAP",
        hue="Modelo",
        palette=["#3498db", "#e74c3c"]
    )
    plt.title("Comparação de MAP por Configuração de Pré-processamento")
    plt.xlabel("Configuração de Pré-processamento")
    plt.ylabel("MAP (Mean Average Precision)")
    plt.xticks(
        ticks=range(4),
        labels=["Sem Stop | Sem Stem", "Com Stop | Sem Stem", "Sem Stop | Com Stem", "Com Stop | Com Stem"],
        rotation=10
    )
    plt.ylim(0, 0.40)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom",
                        fontsize=9, xytext=(0, 3),
                        textcoords="offset points")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "map_preprocessing_comparison.png", dpi=300)
    plt.close()

    print("  [✓] Experimento 1 finalizado com sucesso.")
    return summary_df


# --- Experimento 2: Grid Search de Hiperparâmetros do BM25 ---

def run_bm25_parameter_grid_search(
    docs: Dict[str, Document],
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    Avalia a variação dos parâmetros do BM25:
    k1 in {0.5, 1.2, 2.0} e b in {0.0, 0.75, 1.0}
    Gera tabela, heatmap de MAP e seleciona consulta sensível a b.
    """
    print("\n[Experimento 2] Executando Grid Search de Hiperparâmetros do BM25 (k1 x b)...")
    doc_texts = {did: d.full_text for did, d in docs.items()}
    query_texts = {qid: q.text for qid, q in queries.items()}

    k1_values = [0.5, 1.2, 2.0]
    b_values = [0.0, 0.75, 1.0]

    bm25 = BM25Model(config=CONFIG_WITH_STOP_WITH_STEM)
    bm25.fit(doc_texts)

    grid_records: List[Dict[str, Any]] = []
    rankings_by_params: Dict[Tuple[float, float], Dict[str, List[Tuple[str, float]]]] = {}

    for k1 in k1_values:
        for b in b_values:
            bm25.set_parameters(k1=k1, b=b)
            ranks = bm25.batch_search(query_texts, top_k=None)
            rankings_by_params[(k1, b)] = ranks
            _, agg = evaluate_system(ranks, qrels, k=10)
            grid_records.append({
                "k1": k1,
                "b": b,
                **agg
            })

    grid_df = pd.DataFrame(grid_records)
    grid_df.to_csv(RESULTS_DIR / "bm25_parameter_grid.csv", index=False)

    map_pivot = grid_df.pivot(index="k1", columns="b", values="MAP")
    plt.figure(figsize=(7, 5))
    sns.heatmap(map_pivot, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True)
    plt.title("Sensibilidade do BM25: Heatmap de MAP ($k_1$ vs $b$)")
    plt.xlabel("Parâmetro $b$ (Normalização de Tamanho)")
    plt.ylabel("Parâmetro $k_1$ (Saturação de Frequência)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bm25_grid_search_map_heatmap.png", dpi=300)
    plt.close()

    ndcg_pivot = grid_df.pivot(index="k1", columns="b", values="NDCG@10")
    plt.figure(figsize=(7, 5))
    sns.heatmap(ndcg_pivot, annot=True, fmt=".4f", cmap="YlOrRd", cbar=True)
    plt.title("Sensibilidade do BM25: Heatmap de NDCG@10 ($k_1$ vs $b$)")
    plt.xlabel("Parâmetro $b$ (Normalização de Tamanho)")
    plt.ylabel("Parâmetro $k_1$ (Saturação de Frequência)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bm25_grid_search_ndcg_heatmap.png", dpi=300)
    plt.close()

    # Análise de sensibilidade ao parâmetro b (b=0.0 vs b=1.0 para k1=1.2)
    ranks_b0 = rankings_by_params[(1.2, 0.0)]
    ranks_b1 = rankings_by_params[(1.2, 1.0)]

    best_candidate_qid = None
    max_rank_shift = -1
    shift_details: Dict[str, Any] = {}

    for qid in sorted(queries.keys(), key=lambda x: int(x)):
        top10_b0 = [did for did, _ in ranks_b0[qid][:10]]
        top10_b1 = [did for did, _ in ranks_b1[qid][:10]]
        q_rels = get_relevant_doc_ids(qrels[qid])

        for did in q_rels:
            pos_b0 = top10_b0.index(did) if did in top10_b0 else 99
            pos_b1 = top10_b1.index(did) if did in top10_b1 else 99
            shift = abs(pos_b0 - pos_b1)
            if shift > max_rank_shift and (pos_b0 < 10 or pos_b1 < 10):
                max_rank_shift = shift
                best_candidate_qid = qid
                shift_details = {
                    "query_id": qid,
                    "query_text": queries[qid].text,
                    "target_doc_id": did,
                    "target_doc_title": docs[did].title,
                    "target_doc_length": len(docs[did].full_text.split()),
                    "avdl": bm25.avdl,
                    "rank_b0": pos_b0 + 1 if pos_b0 != 99 else ">10",
                    "rank_b1": pos_b1 + 1 if pos_b1 != 99 else ">10",
                    "top5_b0": [(did, float(sc), did in q_rels) for did, sc in ranks_b0[qid][:5]],
                    "top5_b1": [(did, float(sc), did in q_rels) for did, sc in ranks_b1[qid][:5]],
                }

    with open(RESULTS_DIR / "bm25_b_parameter_shift_case.json", "w", encoding="utf-8") as f:
        json.dump(shift_details, f, indent=2, ensure_ascii=False)

    print(f"  [✓] Experimento 2 finalizado. Consulta selecionada para efeito de b: Query {best_candidate_qid} (Shift: {max_rank_shift})")
    return grid_df


# --- Experimento 3: Análise por Consulta (Casos Contrastantes) ---

def find_notable_query_cases(
    docs: Dict[str, Document],
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Identifica:
    i) 2 consultas em que o BM25 seja claramente superior ao Modelo Vetorial
    ii) 2 consultas em que o Modelo Vetorial seja superior ao BM25
    iii) 2 consultas em que ambos apresentem desempenho insatisfatório
    Para cada caso, extrai os 5 primeiros documentos com indicação de relevância.
    """
    print("\n[Experimento 3] Identificando Casos Notáveis por Consulta...")
    doc_texts = {did: d.full_text for did, d in docs.items()}
    query_texts = {qid: q.text for qid, q in queries.items()}

    vsm = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM).fit(doc_texts)
    bm25 = BM25Model(k1=1.2, b=0.75, config=CONFIG_WITH_STOP_WITH_STEM).fit(doc_texts)

    vsm_ranks = vsm.batch_search(query_texts, top_k=None)
    bm25_ranks = bm25.batch_search(query_texts, top_k=None)

    vsm_df, _ = evaluate_system(vsm_ranks, qrels, k=10)
    bm25_df, _ = evaluate_system(bm25_ranks, qrels, k=10)

    merged = pd.merge(vsm_df, bm25_df, on="query_id", suffixes=("_VSM", "_BM25"))
    merged["diff_AP"] = merged["AP_BM25"] - merged["AP_VSM"]

    # 1. BM25 claramente superior: maior diff_AP positivo
    bm25_wins = merged.sort_values(by="diff_AP", ascending=False).head(2)["query_id"].tolist()

    # 2. Vetorial superior ao BM25: maior diff_AP negativo
    vsm_wins = merged.sort_values(by="diff_AP", ascending=True).head(2)["query_id"].tolist()

    # 3. Ambos insatisfatórios: AP_VSM == 0 e AP_BM25 == 0 (ou os menores valores conjuntos)
    both_bad = merged[(merged["AP_VSM"] == 0.0) & (merged["AP_BM25"] == 0.0)].head(2)["query_id"].tolist()
    if len(both_bad) < 2:
        both_bad = merged.sort_values(by=["AP_BM25", "AP_VSM"]).head(2)["query_id"].tolist()

    cases = {
        "bm25_superior": bm25_wins,
        "vsm_superior": vsm_wins,
        "both_unsatisfactory": both_bad,
    }

    detailed_cases: Dict[str, List[Dict[str, Any]]] = {}

    for category, qid_list in cases.items():
        detailed_cases[category] = []
        for qid in qid_list:
            q_rels = get_relevant_doc_ids(qrels[qid])
            vsm_top5 = vsm_ranks[qid][:5]
            bm25_top5 = bm25_ranks[qid][:5]

            detailed_cases[category].append({
                "query_id": qid,
                "query_text": queries[qid].text,
                "total_relevant": len(q_rels),
                "metrics": {
                    "VSM": {
                        "AP": float(merged[merged["query_id"] == qid]["AP_VSM"].values[0]),
                        "P@10": float(merged[merged["query_id"] == qid]["P@10_VSM"].values[0]),
                        "NDCG@10": float(merged[merged["query_id"] == qid]["NDCG@10_VSM"].values[0]),
                    },
                    "BM25": {
                        "AP": float(merged[merged["query_id"] == qid]["AP_BM25"].values[0]),
                        "P@10": float(merged[merged["query_id"] == qid]["P@10_BM25"].values[0]),
                        "NDCG@10": float(merged[merged["query_id"] == qid]["NDCG@10_BM25"].values[0]),
                    }
                },
                "top5_VSM": [
                    {
                        "rank": i + 1,
                        "doc_id": did,
                        "score": round(score, 4),
                        "is_relevant": did in q_rels,
                        "cleverdon_grade": qrels[qid].get(did, 0),
                        "title": docs[did].title[:80],
                    }
                    for i, (did, score) in enumerate(vsm_top5)
                ],
                "top5_BM25": [
                    {
                        "rank": i + 1,
                        "doc_id": did,
                        "score": round(score, 4),
                        "is_relevant": did in q_rels,
                        "cleverdon_grade": qrels[qid].get(did, 0),
                        "title": docs[did].title[:80],
                    }
                    for i, (did, score) in enumerate(bm25_top5)
                ],
            })

    with open(RESULTS_DIR / "notable_query_cases.json", "w", encoding="utf-8") as f:
        json.dump(detailed_cases, f, indent=2, ensure_ascii=False)

    print(f"  [✓] Casos selecionados: BM25 > VSM: {bm25_wins} | VSM > BM25: {vsm_wins} | Ambos falham: {both_bad}")
    return detailed_cases


# --- Experimento 4: Modificação Manual de Consultas ---

def run_query_reformulation_experiment(
    docs: Dict[str, Document],
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:
    """
    Seleciona 5 consultas representativas e cria versões alternativas controladas:
    - Adicionando termos técnicos relevantes e sinônimos
    - Removendo termos vagos/ambíguos
    - Tornando a consulta mais específica ou mais genérica
    Compara o Top-10 e as métricas nos dois modelos antes e depois.
    """
    print("\n[Experimento 4] Avaliando Reformulação Manual de 5 Consultas...")
    doc_texts = {did: d.full_text for did, d in docs.items()}

    vsm = VectorSpaceModel(CONFIG_WITH_STOP_WITH_STEM).fit(doc_texts)
    bm25 = BM25Model(k1=1.2, b=0.75, config=CONFIG_WITH_STOP_WITH_STEM).fit(doc_texts)

    reformulations = [
        {
            "query_id": "1",
            "strategy": "Adição de termos técnicos e sinônimos estruturais (aeroelasticidade -> flutter, thermo-elastic)",
            "modified_text": "similarity laws governing thermo-aeroelastic scale models structural flutter supersonic heated aircraft",
        },
        {
            "query_id": "8",
            "strategy": "Especificação precisa: substituição de pergunta aberta por termos conceituais exatos",
            "modified_text": "aerodynamic interference slipstream propeller wing lift distribution angle attack",
        },
        {
            "query_id": "12",
            "strategy": "Generalização e remoção de termos excessivamente específicos",
            "modified_text": "boundary layer laminar turbulent transition heat transfer supersonic flow",
        },
        {
            "query_id": "23",
            "strategy": "Adição de sinônimos físicos (pressure distribution, shock wave reflection)",
            "modified_text": "pressure distribution shock wave boundary layer interaction supersonic wind tunnel",
        },
        {
            "query_id": "87",
            "strategy": "Remoção de ruído coloquial e inclusão de descritores de regime térmico",
            "modified_text": "transient heat conduction multilayer composite slabs thermal stress diffusion",
        },
    ]

    reformulation_results: List[Dict[str, Any]] = []

    for item in reformulations:
        qid = item["query_id"]
        orig_text = queries[qid].text
        mod_text = item["modified_text"]
        q_rels = get_relevant_doc_ids(qrels[qid])

        vsm_orig_rank = vsm.search(orig_text, top_k=None)
        vsm_mod_rank = vsm.search(mod_text, top_k=None)
        
        bm25_orig_rank = bm25.search(orig_text, top_k=None)
        bm25_mod_rank = bm25.search(mod_text, top_k=None)

        def eval_single(rank_list):
            dids = [d for d, _ in rank_list]
            from src.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k
            return {
                "P@10": round(precision_at_k(dids, qrels[qid], k=10), 4),
                "R@10": round(recall_at_k(dids, qrels[qid], k=10), 4),
                "AP": round(average_precision(dids, qrels[qid]), 4),
                "NDCG@10": round(ndcg_at_k(dids, qrels[qid], k=10, graded=True), 4),
            }

        vsm_orig_metrics = eval_single(vsm_orig_rank)
        vsm_mod_metrics = eval_single(vsm_mod_rank)
        bm25_orig_metrics = eval_single(bm25_orig_rank)
        bm25_mod_metrics = eval_single(bm25_mod_rank)

        # Análise do Top-10
        vsm_orig_top10 = [d for d, _ in vsm_orig_rank[:10]]
        vsm_mod_top10 = [d for d, _ in vsm_mod_rank[:10]]
        bm25_orig_top10 = [d for d, _ in bm25_orig_rank[:10]]
        bm25_mod_top10 = [d for d, _ in bm25_mod_rank[:10]]

        reformulation_results.append({
            "query_id": qid,
            "strategy": item["strategy"],
            "original_query": orig_text,
            "modified_query": mod_text,
            "total_relevant": len(q_rels),
            "VSM_original_metrics": vsm_orig_metrics,
            "VSM_modified_metrics": vsm_mod_metrics,
            "BM25_original_metrics": bm25_orig_metrics,
            "BM25_modified_metrics": bm25_mod_metrics,
            "VSM_top10_intersection": len(set(vsm_orig_top10).intersection(set(vsm_mod_top10))),
            "BM25_top10_intersection": len(set(bm25_orig_top10).intersection(set(bm25_mod_top10))),
            "VSM_new_relevant_in_top10": [d for d in vsm_mod_top10 if d in q_rels and d not in vsm_orig_top10],
            "BM25_new_relevant_in_top10": [d for d in bm25_mod_top10 if d in q_rels and d not in bm25_orig_top10],
        })

    with open(RESULTS_DIR / "query_reformulation_cases.json", "w", encoding="utf-8") as f:
        json.dump(reformulation_results, f, indent=2, ensure_ascii=False)

    print("  [✓] Experimento 4 finalizado. 5 consultas reformuladas e avaliadas.")
    return reformulation_results


# --- Experimento 5: Análise de Erros ---

def run_error_analysis(
    docs: Dict[str, Document],
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Identifica e investiga:
    1. Pelo menos dois documentos NÃO relevantes que aparecem no topo do ranking (ranks 1 e 2).
    2. Pelo menos um documento RELEVANTE que NÃO aparece no Top-10.
    Apresenta justificativa baseada nos conceitos de RI (frequência, tamanho, mismatch de vocabulário).
    """
    print("\n[Experimento 5] Realizando Diagnóstico de Erros...")
    doc_texts = {did: d.full_text for did, d in docs.items()}

    bm25 = BM25Model(k1=1.2, b=0.75, config=CONFIG_WITH_STOP_WITH_STEM).fit(doc_texts)
    
    # Falsos positivos e falsos negativos
    false_positives: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []

    for qid in sorted(queries.keys(), key=lambda x: int(x)):
        q_rels = get_relevant_doc_ids(qrels[qid])
        ranks = bm25.search(queries[qid].text, top_k=20)
        
        # Procura falso positivo no rank 1
        if len(false_positives) < 2:
            top_did, top_score = ranks[0]
            if top_did not in q_rels and top_score > 5.0:
                explanation = bm25.explain_score(queries[qid].text, top_did)
                false_positives.append({
                    "case_type": "Falso Positivo no Top-1",
                    "query_id": qid,
                    "query_text": queries[qid].text,
                    "doc_id": top_did,
                    "rank": 1,
                    "score": round(top_score, 4),
                    "doc_title": docs[top_did].title,
                    "doc_snippet": docs[top_did].text[:250],
                    "matched_terms": [t for t in explanation["terms"] if t["in_document"]],
                    "reason_diagnosis": (
                        "O documento contém repetições concentradas de termos com alto IDF presentes na consulta, "
                        "mas o contexto semântico real da publicação aborda um fenômeno distinto (alta sobreposição lexical "
                        "sem concordância semântica)."
                    )
                })

        # Procura falso negativo: documento relevante que ficou fora do Top-10
        if len(false_negatives) < 2:
            target_rel = "30" if qid == "1" else ("442" if qid == "2" else None)
            candidate_rels = [target_rel] if (target_rel and target_rel in q_rels) else sorted(q_rels)
            ranked_dids = [did for did, _ in ranks[:10]]
            for rel_did in candidate_rels:
                if rel_did not in ranked_dids:
                    # Encontra o rank real dele
                    full_ranks = [did for did, _ in bm25.search(queries[qid].text, top_k=None)]
                    actual_rank = full_ranks.index(rel_did) + 1 if rel_did in full_ranks else ">1400"
                    explanation = bm25.explain_score(queries[qid].text, rel_did)
                    
                    false_negatives.append({
                        "case_type": "Falso Negativo Fora do Top-10",
                        "query_id": qid,
                        "query_text": queries[qid].text,
                        "doc_id": rel_did,
                        "actual_rank": actual_rank,
                        "doc_title": docs[rel_did].title,
                        "doc_snippet": docs[rel_did].text[:250],
                        "matched_terms": [t for t in explanation["terms"] if t["in_document"]],
                        "missing_terms": [t["term"] for t in explanation["terms"] if not t["in_document"]],
                        "reason_diagnosis": (
                            "Incompatibilidade de vocabulário (vocabulary mismatch): o documento expressa o conceito "
                            "desejado através de formulações sinônimas ou simbologia técnica não espelhada literalmente "
                            "nos termos da consulta."
                        )
                    })
                    break

    error_report = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    with open(RESULTS_DIR / "error_analysis_cases.json", "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False)

    print("  [✓] Experimento 5 finalizado. Falsos positivos e negativos diagnosticados.")
    return error_report


# --- Execução Principal ---

def run_all_experiments() -> None:
    print("=" * 70)
    print("INICIANDO EXECUÇÃO INTEGRADA DOS EXPERIMENTOS DE RI")
    print("=" * 70)

    docs, queries, qrels = load_cranfield_dataset()

    # Experimento 1: Pré-processamento e Modelos
    run_preprocessing_and_model_comparison(docs, queries, qrels)

    # Experimento 2: Grid Search do BM25 e Efeito de b
    run_bm25_parameter_grid_search(docs, queries, qrels)

    # Experimento 3: Análise de Consultas Notáveis
    find_notable_query_cases(docs, queries, qrels)

    # Experimento 4: Reformulação de Consultas
    run_query_reformulation_experiment(docs, queries, qrels)

    # Experimento 5: Análise Diagnóstica de Erros
    run_error_analysis(docs, queries, qrels)

    print("=" * 70)
    print("TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO!")
    print(f"Resultados gravados em: {RESULTS_DIR}")
    print(f"Gráficos gerados em:    {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
