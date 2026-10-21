"""Question 2A(a)(iii): controlled RotatE optimizer / learning-rate experiments.

Run this file from the assignment project root, e.g.:
    python train_rotate_2a_optimization.py --optimizer adam --lr 0.0005

The existing 100-dimensional, Adam 0.001 checkpoint is the baseline.
All experiments evaluate on the validation split only.
"""

import argparse
import time
from pathlib import Path

import torch
from pykeen.datasets import FB15k237
from pykeen.losses import MarginRankingLoss
from pykeen.pipeline import pipeline


parser = argparse.ArgumentParser(description="RotatE training optimization experiment")
parser.add_argument("--optimizer", choices=("adam", "adamw"), required=True)
parser.add_argument("--lr", type=float, required=True)
args = parser.parse_args()

if args.lr <= 0:
    parser.error("--lr must be positive")

optimizer_classes = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.Adam,
}

optimizer_kwargs = {"lr": args.lr}

if args.optimizer == "adamw":
    optimizer_kwargs["weight_decay"] = 0.01
    optimizer_kwargs["decoupled_weight_decay"] = True

output_dir = Path("outputs") / (
    f"rotate_{args.optimizer}_lr{args.lr:g}_50epochs".replace(".", "p")
)
if output_dir.exists():
    parser.error(f"Output directory already exists: {output_dir}. Choose another run or move it first.")

dataset = FB15k237()
print("Training triples:", dataset.training.num_triples)
print("Validation triples:", dataset.validation.num_triples)
print("Testing triples:", dataset.testing.num_triples)
print(f"Experiment: optimizer={args.optimizer}, learning_rate={args.lr}")
print(f"Optimizer arguments: {optimizer_kwargs}")

start = time.perf_counter()
result = pipeline(
    dataset=dataset,
    model="RotatE",
    model_kwargs={"embedding_dim": 100},
    loss=MarginRankingLoss,
    loss_kwargs={"margin": 1.0},
    optimizer=optimizer_classes[args.optimizer],
    optimizer_kwargs=optimizer_kwargs,
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
elapsed = time.perf_counter() - start

result.save_to_directory(str(output_dir))
mrr = result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")

print(f"Validation filtered MRR: {mrr:.4f}")
print(f"Elapsed time: {elapsed / 60:.1f} minutes")
print(f"Saved model and results to: {output_dir}")
