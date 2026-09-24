# SCC0282 — Recuperação de Informação (2026)
## Trabalho Prático 1: Implementação e Avaliação de Modelos de RI

**Instituto de Ciências Matemáticas e de Computação — Universidade de São Paulo (ICMC-USP)**  
**Docente:** Prof. Dr. Marcelo Garcia Manzato  

---

## 1. Integrantes do Grupo

| Nome Completo | Número USP | E-mail Institucional |
| :--- | :--- | :--- |
| Ana Rita Marega Gonçalves | 15746365 | anamarega@usp.br |

---

## 2. Identificação e Forma de Obtenção da Base de Dados

- **Identificação da Base:** Coleção de Teste **Cranfield** (*Cranfield Collection*), coleção de referência clássica na área de Recuperação de Informação composta por:
  - **1.400 documentos** técnicos da área de aerodinâmica e termodinâmica (`cran.all.1400`).
  - **225 consultas** textuais formuladas por pesquisadores (`cran.qry`).
  - **Julgamentos de relevância (*qrels*)** para as 225 consultas (`cranqrel`), avaliados por especialistas na escala Cleverdon de 1 a 4 (relevantes) e -1 (não relevante).

- **Forma de Obtenção:**
  - Os arquivos brutos da coleção já se encontram organizados e prontos para uso no diretório local `data/raw/` do repositório.
  - Caso seja necessário reobter ou verificar os arquivos originais diretamente dos servidores canônicos da Universidade de Glasgow, basta executar o módulo de carregamento:
    ```bash
    python3 src/dataset.py
    ```
    O script verifica os arquivos locais e, caso ausentes, realiza o download automático do pacote oficial (`http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz`), realizando a extração e a validação do alinhamento entre consultas e qrels. Também é suportado o carregamento via biblioteca `ir-datasets` (`irds/cranfield`).

---

## 3. Versão da Linguagem e Principais Bibliotecas Utilizadas

- **Versão da Linguagem:** Python 3.10 ou superior (desenvolvido e homologado em **Python 3.13.x**).
- **Principais Bibliotecas Utilizadas:**
  - `nltk` (>= 3.9.1): Tokenização textual (`re`), catálogo de *stopwords* em língua inglesa e estematização morfológica (*Porter Stemmer*).
  - `scikit-learn` (>= 1.5.0): Vetorização esparsa com TF-IDF e cálculo matricial da similaridade de cosseno com normalização Euclidiana $L_2$.
  - `numpy` (>= 2.0.0): Operações numéricas e manipulação de vetores de pesos.
  - `pandas` (>= 2.2.0): Estruturação de dados tabulares, cálculo de métricas por consulta e agregação de médias globais.
  - `matplotlib` (>= 3.9.0) e `seaborn` (>= 0.13.0): Geração de gráficos acadêmicos em alta resolução (mapas de calor do *grid search* e gráficos de barras comparativos).
  - `pytest` (>= 8.0.0): Suíte de testes unitários para verificação matemática e funcional dos componentes.

---

## 4. Instruções de Instalação e Execução

### 4.1. Instalação das Dependências

Recomenda-se a utilização de um ambiente virtual dedicado:

```bash
# 1. Crie o ambiente virtual
python3 -m venv .venv

# 2. Ative o ambiente
source .venv/bin/activate  # No Linux/macOS
# .venv\Scripts\activate   # No Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

### 4.2. Execução dos Testes Unitários

Para executar a bateria de testes unitários e verificar a corretude das métricas, do pré-processamento, do Modelo Vetorial e do BM25:

```bash
pytest
```
*Total esperado: 21 testes aprovados (`21 passed`).*

### 4.3. Execução dos Experimentos

Para reproduzir todos os experimentos práticos, gerar as tabelas em `results/` e os gráficos em `figures/`:

```bash
python3 src/experiments.py
```

O script executa em sequência:
1. **Comparação entre Modelos e Pré-processamento:** Avalia o Modelo Vetorial (VSM) e o BM25 nas 4 configurações de vocabulário sobre as 225 consultas.
2. **Ajuste de Parâmetros do BM25:** *Grid search* $3 \times 3$ variando $k_1 \in \{0.5, 1.2, 2.0\}$ e $b \in \{0.0, 0.75, 1.0\}$, gerando os mapas de calor e análise do impacto do fator $b$.
3. **Análise por Consulta:** Diagnóstico de consultas com comportamentos contrastantes (BM25 superior, VSM superior e casos de baixa eficácia conjunta).
4. **Reformulação Manual de Consultas:** Teste controlado de estratégias de reformulação em 5 consultas da base.
5. **Diagnóstico de Erros:** Análise termo a termo de falsos positivos nas primeiras posições e falsos negativos fora do Top-10.

---

## 5. Estrutura do Repositório

```text
.
├── data/
│   └── raw/                       # Arquivos da coleção Cranfield (cran.all.1400, cran.qry, cranqrel)
├── figures/                       # Gráficos e mapas de calor gerados pelos experimentos
│   ├── map_preprocessing_comparison.png
│   ├── bm25_grid_search_map_heatmap.png
│   └── bm25_grid_search_ndcg_heatmap.png
├── results/                       # Tabelas e arquivos brutos de resultado (CSV e JSON)
│   ├── per_query_comparison_default_config.csv    # Resultados detalhados por consulta (225 queries)
│   ├── preprocessing_and_models_comparison.csv    # Comparativo global dos 4 pré-processamentos
│   ├── bm25_parameter_grid.csv                    # Tabela do grid search k1 x b
│   ├── bm25_b_parameter_shift_case.json           # Caso de estudo do parâmetro b (Q50 / Doc 326)
│   ├── notable_query_cases.json                   # Casos notáveis (BM25 > VSM, VSM > BM25, falhas)
│   ├── query_reformulation_cases.json             # 5 consultas reformuladas (Top-10 antes e depois)
│   └── error_analysis_cases.json                  # Diagnóstico de falsos positivos e negativos
├── src/                           # Código-fonte modular do sistema
│   ├── dataset.py                 # Parser e carregador da coleção Cranfield
│   ├── preprocessor.py            # Tokenização, normalização, stopwords e Porter Stemmer
│   ├── vector_model.py            # Modelo Vetorial (TF-IDF e Similaridade do Cosseno)
│   ├── bm25.py                    # Modelo BM25 (Robertson-Zaragoza) e decomposição de score
│   ├── metrics.py                 # Métricas de avaliação (P@k, R@k, MAP, NDCG@k, MRR)
│   └── experiments.py             # Pipeline completo de execução experimental
├── tests/                         # Bateria de testes unitários automatizados
│   ├── test_dataset.py
│   ├── test_preprocessor.py
│   ├── test_vector_model.py
│   ├── test_bm25.py
│   └── test_metrics.py
├── pytest.ini                     # Configuração do pytest
├── requirements.txt               # Lista de dependências do projeto
└── README.md                      # Documentação técnica do projeto
```

---

## 6. Resultados Completos dos Experimentos e Dados por Consulta

Todos os dados brutos e arquivos intermediários que originaram as tabelas e gráficos do relatório estão salvos e acessíveis no repositório:

### 6.1. Arquivos de Métricas e Dados por Consulta (Diretório `results/`)

- **Resultados Detalhados por Consulta (225 Consultas):** [`results/per_query_comparison_default_config.csv`](results/per_query_comparison_default_config.csv)  
  Contém as métricas individuais ($P@10$, $R@10$, $AP$, $NDCG@10$, $MRR$ e a diferença $AP_{BM25} - AP_{VSM}$) para **cada uma das 225 consultas da base Cranfield**, confrontando diretamente o Modelo Vetorial e o BM25 na configuração completa ($C_4$).
- **Comparativo Consolidado de Modelos e Pré-processamentos:** [`results/preprocessing_and_models_comparison.csv`](results/preprocessing_and_models_comparison.csv)  
  Tabela completa com as médias agregadas para as 4 configurações de pré-processamento em ambos os modelos.
- **Grade Paramétrica do BM25 (*Grid Search*):** [`results/bm25_parameter_grid.csv`](results/bm25_parameter_grid.csv)  
  Dados brutos da varredura $3 \times 3$ combinando $k_1 \in \{0.5, 1.2, 2.0\}$ e $b \in \{0.0, 0.75, 1.0\}$.
- **Estudos de Casos Específicos e Diagnóstico Termo a Termo:**
  - `results/bm25_b_parameter_shift_case.json`: Diagnóstico do impacto do parâmetro $b$ na Consulta 50 e Documento 326.
  - `results/notable_query_cases.json`: Top-5 documentos retornados para os casos de maior discrepância entre os modelos.
  - `results/query_reformulation_cases.json`: Avaliação do Top-10 e métricas antes e depois da reformulação de 5 consultas.
  - `results/error_analysis_cases.json`: Auditoria de pesos TF e IDF para falsos positivos no Top-1 e falsos negativos fora do Top-10.

### 6.2. Gráficos e Mapas de Calor (Diretório `figures/`)

- **Comparação de MAP entre Configurações de Pré-processamento:** [`figures/map_preprocessing_comparison.png`](figures/map_preprocessing_comparison.png)
- **Mapa de Calor de MAP (*Grid Search* $k_1 \times b$):** [`figures/bm25_grid_search_map_heatmap.png`](figures/bm25_grid_search_map_heatmap.png)
- **Mapa de Calor de NDCG@10 (*Grid Search* $k_1 \times b$):** [`figures/bm25_grid_search_ndcg_heatmap.png`](figures/bm25_grid_search_ndcg_heatmap.png)

---

## 7. Tabela Resumo dos Resultados Principais

Resultados médios obtidos sobre as 225 consultas da base Cranfield com corte em $k=10$:

| Modelo | Pré-processamento | P@10 | R@10 | MAP | NDCG@10 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vetorial (VSM)** | Sem Stopwords / Sem Stemming | 0.2284 | 0.3766 | 0.2785 | 0.3325 | 0.5082 |
| **Vetorial (VSM)** | Com Stopwords / Sem Stemming | 0.2244 | 0.3720 | 0.2845 | 0.3291 | 0.5177 |
| **Vetorial (VSM)** | Sem Stopwords / Com Stemming | 0.2324 | 0.3822 | 0.2995 | 0.3441 | 0.5213 |
| **Vetorial (VSM)** | **Com Stopwords / Com Stemming** | **0.2422** | **0.3997** | **0.3069** | **0.3540** | **0.5352** |
| **BM25** ($k_1=1.2, b=0.75$) | Sem Stopwords / Sem Stemming | 0.2253 | 0.3829 | 0.2781 | 0.3337 | 0.5010 |
| **BM25** ($k_1=1.2, b=0.75$) | Com Stopwords / Sem Stemming | 0.2324 | 0.3949 | 0.2941 | 0.3505 | 0.5300 |
| **BM25** ($k_1=1.2, b=0.75$) | Sem Stopwords / Com Stemming | 0.2324 | 0.3949 | 0.3045 | 0.3573 | 0.5371 |
| **BM25** ($k_1=1.2, b=0.75$) | **Com Stopwords / Com Stemming** | **0.2404** | **0.4050** | **0.3148** | **0.3619** | **0.5423** |

*No ajuste fino de parâmetros (grid search), a combinação $k_1 = 2.0$ e $b = 0.75$ alcançou o melhor resultado global de MAP (0.3207) e NDCG@10 (0.3740).*
