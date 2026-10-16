
from pathlib import Path
import csv

import torch
from pykeen.datasets import FB15k237
from pykeen.evaluation import RankBasedEvaluator


# 1. Load the FB15k-237 dataset
dataset = FB15k237()

training = dataset.training
validation = dataset.validation
testing = dataset.testing

print("Testing triples:", testing.num_triples)


# 2. Paths to the saved trained models
MODEL_PATHS = {
    "TransE": "outputs/transe_50epochs/trained_model.pkl",
    "RotatE": "outputs/rotate_50epochs/trained_model.pkl",
    "RESCAL": "outputs/rescal_reg001_10epochs/trained_model.pkl",
}

# Directory to save final results
OUTPUT_DIR = Path("report/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_table = []


# 3. Evaluate each model
for model_name, model_path in MODEL_PATHS.items():

    print(f"\nEvaluating {model_name}...")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load the previously trained model
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    # Set the model to evaluation mode
    model.eval()

    # 4. Create filtered rank-based evaluator
    evaluator = RankBasedEvaluator(
        filtered=True,
    )

    # 5. Evaluate on the TEST dataset
    metric_results = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,

        # Filter out other known true triples
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples,
        ],

        batch_size=32,
        device=torch.device("cpu"),
    )

    # 6. Get filtered MRR
    filtered_mrr = metric_results.get_metric(
        "both.realistic.inverse_harmonic_mean_rank"
    )

    # Also record tail-prediction MRR
    tail_mrr = metric_results.get_metric(
        "tail.realistic.inverse_harmonic_mean_rank"
    )

    print(f"{model_name} Filtered MRR: {filtered_mrr:.4f}")
    print(f"{model_name} Tail MRR: {tail_mrr:.4f}")

    results_table.append({
        "model": model_name,
        "filtered_mrr": filtered_mrr,
        "tail_mrr": tail_mrr,
    })


# 7. Save the results to CSV
output_path = OUTPUT_DIR / "q1c_test_mrr.csv"

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "filtered_mrr", "tail_mrr"],
    )

    writer.writeheader()
    writer.writerows(results_table)

print("\nFinal Test Results:")
for row in results_table:
    print(
        f"{row['model']}: "
        f"MRR = {row['filtered_mrr']:.4f}"
    )

print(f"\nResults saved to {output_path}")
