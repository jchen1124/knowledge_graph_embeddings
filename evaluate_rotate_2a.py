"""Question 2A(b): final filtered test evaluation of selected RotatE model.

Run from the assignment repository root:
    python evaluate_rotate_2a.py

This evaluates an already-trained checkpoint and does not train or tune it.
The test result is saved separately from the Question 1 results.
"""

from pathlib import Path
import csv

import torch
from pykeen.datasets import FB15k237
from pykeen.evaluation import RankBasedEvaluator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "rotate_adamw_lr0p0005_50epochs"
    / "trained_model.pkl"
)
OUTPUT_DIR = BASE_DIR / "report" / "results"
OUTPUT_PATH = OUTPUT_DIR / "q2a_selected_rotate_test_mrr.csv"


# 1. Load the same standard FB15k-237 splits used in Question 1.
dataset = FB15k237()
training = dataset.training
validation = dataset.validation
testing = dataset.testing
print(f"Testing triples: {testing.num_triples}")


# 2. Load the best configuration, chosen using validation MRR (0.2410).
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Saved checkpoint not found: {MODEL_PATH}")

print(f"Loading selected RotatE checkpoint: {MODEL_PATH}")
# Use weights_only=False only for a checkpoint you created and trust.
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()


# 3. Evaluate on the held-out TEST split, using the exact same filtering
#    approach and metric keys as the original evaluate_models.py script.
evaluator = RankBasedEvaluator(filtered=True)
metric_results = evaluator.evaluate(
    model=model,
    mapped_triples=testing.mapped_triples,
    additional_filter_triples=[
        training.mapped_triples,
        validation.mapped_triples,
    ],
    batch_size=32,
    device=torch.device("cpu"),
)

filtered_mrr = metric_results.get_metric(
    "both.realistic.inverse_harmonic_mean_rank"
)
tail_mrr = metric_results.get_metric(
    "tail.realistic.inverse_harmonic_mean_rank"
)


# 4. Preserve Question 1's results by writing to a separate CSV.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "filtered_mrr", "tail_mrr"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "model": "RotatE (AdamW, lr=0.0005, wd=0.01)",
            "filtered_mrr": filtered_mrr,
            "tail_mrr": tail_mrr,
        }
    )

print("\n===== QUESTION 2A(b): FINAL TEST RESULT =====")
print(f"Filtered test MRR (both): {filtered_mrr:.4f}")
print(f"Filtered test MRR (tail): {tail_mrr:.4f}")
print(f"Saved results to: {OUTPUT_PATH}")
