# CMSC 491 — Assignment 2: Knowledge Graph Embeddings and Question Answering

This repository contains the Python code for **Question 1 (Knowledge Graph Embeddings)**, **Question 2A (KGE Improvement Experiments)**, and **Question 3 (Knowledge Graph Question Answering)**. The scripts for Questions 1 and 2A are in the **project root**, while the scripts for Question 3 are inside **`Question_3/`**. The written report contains the explanations and analysis.

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

### Question 2A — Improving KGE Performance (FB15k-237)

| File | Assignment part | Purpose |
|---|---|---|
| `train_rotate_2a.py` | **2A(a)(i)** | Compares RotatE embedding dimensions of 50 and 200 against the existing 100-dimensional baseline, saving separate checkpoints and validation results. |
| `train_rotate_2a_negative.py` | **2A(a)(ii)** | Trains RotatE with Bernoulli negative sampling for comparison against the basic negative-sampling baseline. |
| `train_rotate_2a_optimization.py` | **2A(a)(iii)** | Tests Adam and AdamW optimizer configurations at different learning rates while keeping the remaining training settings fixed. |
| `evaluate_rotate_2a.py` | **2A(b)** | Loads the selected RotatE checkpoint and evaluates filtered MRR on the held-out FB15k-237 test set without retraining. |

**Question 2A(b)** summarizes the validation comparisons and final test result, while **Question 2A(c)** explains the observed changes in the written report.

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
├── train_rotate_2a.py
├── train_rotate_2a_negative.py
├── train_rotate_2a_optimization.py
├── evaluate_rotate_2a.py
├── Question_3/
│   ├── inspect_dataset.py
│   ├── construct_graph.py
│   ├── train_kge.py
│   ├── kgqa.py
│   ├── evaluate_kgqa.py
│   ├── data/                         # SimpleQuestionsWikidata data splits
│   └── outputs/                      # RDF graph and KGQA TransE checkpoint
├── outputs/                          # Question 1 and 2A checkpoints and metadata
├── report/
│   ├── figures/                      # Question 1 PCA plots
│   └── results/                      # Question 1, 2A, and 3 evaluation outputs
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

### Question 2A

The original 100-dimensional RotatE model (`train_rotate.py`) serves as the baseline. Each experiment saves its checkpoint separately and evaluates configurations using the validation set.

```bash
# 2A(a)(i): Compare embedding dimensions
python train_rotate_2a.py --embedding-dim 50
python train_rotate_2a.py --embedding-dim 200

# 2A(a)(ii): Compare Bernoulli against the existing basic-sampling baseline
python train_rotate_2a_negative.py

# 2A(a)(iii): Compare learning rates and optimizer configurations
python train_rotate_2a_optimization.py --optimizer adam --lr 0.0005
python train_rotate_2a_optimization.py --optimizer adamw --lr 0.001
python train_rotate_2a_optimization.py --optimizer adamw --lr 0.0005

# 2A(b): Evaluate the validation-selected checkpoint on the test set
python evaluate_rotate_2a.py
```

Question 2A checkpoints are saved in separate `outputs/rotate_*_50epochs/` directories. The final test evaluation is written to `report/results/q2a_selected_rotate_test_mrr.csv`, leaving the Question 1 test results unchanged.

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
