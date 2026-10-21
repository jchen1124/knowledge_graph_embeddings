"""Question 2A(a)(i): controlled RotatE embedding-dimension experiment.

Run from the project root:
    python train_rotate_2a.py --embedding-dim 50
    python train_rotate_2a.py --embedding-dim 200

The existing 100-dimensional Question 1 checkpoint is the baseline and is not modified.
"""

import argparse
import time
from pathlib import Path

from pykeen.datasets import FB15k237
from pykeen.losses import MarginRankingLoss
from pykeen.pipeline import pipeline


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Question 2A: RotatE embedding-dimension comparison"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        choices=[50, 200],
        required=True,
        help="Test 50 or 200 dimensions; the 100-dimension baseline is already trained.",
    )
    args = parser.parse_args()

    dim = args.embedding_dim
    output_dir = PROJECT_DIR / "outputs" / f"rotate_dim{dim}_50epochs"

    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}. Refusing to overwrite it.")

    dataset = FB15k237()
    print("Training triples:", dataset.training.num_triples, flush=True)
    print("Validation triples:", dataset.validation.num_triples, flush=True)
    print("Testing triples:", dataset.testing.num_triples, flush=True)
    print(f"RotatE embedding dimension: {dim}", flush=True)
    print("All other settings match train_rotate.py (Question 1).", flush=True)

    start = time.perf_counter()
    result = pipeline(
        dataset=dataset,
        model="RotatE",
        model_kwargs={"embedding_dim": dim},
        loss=MarginRankingLoss,
        loss_kwargs={"margin": 1.0},
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.001},
        training_loop="sLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": 1},
        training_kwargs={"num_epochs": 50, "batch_size": 256},
        use_testing_data=False,
        evaluator_kwargs={"filtered": True},
        evaluation_kwargs={"batch_size": 32},
        random_seed=42,
        device="cpu",
    )

    result.save_to_directory(str(output_dir))
    mrr = result.metric_results.get_metric(
        "both.realistic.inverse_harmonic_mean_rank"
    )

    print(f"\nValidation filtered MRR ({dim} dimensions): {mrr:.4f}")
    print(f"Elapsed time: {(time.perf_counter() - start) / 60:.1f} minutes")
    print(f"Saved model and results to: {output_dir}")
    print("Done. The test set was not used for model selection.")


if __name__ == "__main__":
    main()
