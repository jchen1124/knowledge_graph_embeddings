# CMSC 491 — Assignment 2: Knowledge Graph Embeddings and Question Answering

This repository contains the Python code for **Question 1 (Knowledge Graph Embeddings)** and **Question 3 (Knowledge Graph Question Answering)**. The scripts in the **project root** belong to Question 1, while the scripts inside **`Question_3/`** belong to Question 3.

## File guide

### Question 1 — Knowledge Graph Embeddings (FB15k-237)

| File | Assignment part | Purpose |
|---|---|---|
| `train_transe.py` | **1(a)** | Trains the TransE model using PyKEEN and saves its checkpoint. |
| `train_rotate.py` | **1(a)** | Trains the RotatE model using PyKEEN and saves its checkpoint. |
| `train_rescal.py` | **1(a)** | Trains the RESCAL model using PyKEEN and saves its checkpoint. |
| `visualize_embedding.py` | **1(b)** | Selects entity embeddings and produces two-dimensional PCA plots for the three models. |
| `evaluate_models.py` | **1(c)** | Evaluates saved models on FB15k-237 and calculates filtered MRR. |
| `analyze_rankings.py` | **1(e)** | Examines selected test triples, produces top-five candidate predictions, and records true-answer ranks for each model. |

**Question 1(d)** uses the evaluation and ranking outputs to compare the models; the discussion is in the written report.

### Question 3 — Knowledge Graph Question Answering (SimpleQuestionsWikidata)

| File | Assignment part | Purpose |
|---|---|---|
| `Question_3/inspect_dataset.py` | **3(a)** | Reads and inspects the dataset's question–triple records and splits. |
| `Question_3/construct_graph.py` | **3(a)** | Constructs the training knowledge graph with RDFLib, reports graph statistics, and demonstrates SPARQL queries. |
| `Question_3/train_kge.py` | **3(a)** | Prepares training triples for PyKEEN and trains a separate TransE model for KGQA. |
| `Question_3/kgqa.py` | **3(b)** | Implements the interactive pipeline: identify an entity, classify a relation, construct `(h, r, ?)`, and rank answers using the trained model. |
| `Question_3/evaluate_kgqa.py` | **3(c)–3(d)** | Evaluates the KGQA pipeline on test questions and exports individual predictions for error analysis. |

**Question 3(d)** uses `report/results/q3c_predictions.csv` to investigate failed questions. **Question 3(e)** is the conceptual comparison included in the written report.

## Project directories

```text
.
├── train_transe.py
├── train_rotate.py
├── train_rescal.py
├── visualize_embedding.py
├── evaluate_models.py
├── analyze_rankings.py
├── Question_3/
│   ├── inspect_dataset.py
│   ├── construct_graph.py
│   ├── train_kge.py
│   ├── kgqa.py
│   ├── evaluate_kgqa.py
│   ├── data/                         # SimpleQuestionsWikidata data splits
│   └── outputs/                      # RDF graph and KGQA TransE checkpoint
├── outputs/                          # Question 1 checkpoints and metadata
├── report/
│   ├── figures/                      # Question 1 PCA plots
│   └── results/                      # Evaluation tables and KGQA predictions
├── requirements.txt
└── README.md
```

## Setup

Run the commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project uses PyKEEN, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, and RDFLib. Dependency specifications are in `requirements.txt`.

## Running the code

### Question 1

```bash
# 1(a): Train models
python train_transe.py
python train_rotate.py
python train_rescal.py

# 1(b): Generate PCA plots
python visualize_embedding.py

# 1(c): Evaluate filtered MRR
python evaluate_models.py

# 1(e): Analyze selected triple rankings
python analyze_rankings.py
```

Question 1 outputs include model checkpoints under `outputs/`, plots under `report/figures/`, and result files such as `report/results/q1c_test_mrr.csv` and `report/results/q1e_rankings.csv`.

### Question 3

The dataset files belong in `Question_3/data/`:

```text
annotated_wd_data_train.txt
annotated_wd_data_valid.txt
annotated_wd_data_test.txt
```

Dataset source: [SimpleQuestionsWikidata](https://github.com/askplatypus/wikidata-simplequestions).

```bash
# 3(a): Inspect data, construct graph, and train TransE
python Question_3/inspect_dataset.py
python Question_3/construct_graph.py
python Question_3/train_kge.py

# 3(b): Ask questions interactively
python Question_3/kgqa.py

# 3(c)–3(d): Evaluate and export predictions
python Question_3/evaluate_kgqa.py
```

Question 3 outputs include `Question_3/outputs/training_graph.ttl`, the TransE model and ID mappings under `Question_3/outputs/transe/`, and `report/results/q3c_predictions.csv`. Ensure the model-loading path in `Question_3/kgqa.py` matches the checkpoint location before running the KGQA scripts.

**Note:** The TransE model used for Question 3 is trained on SimpleQuestionsWikidata and is separate from the FB15k-237 TransE model used for Question 1.
