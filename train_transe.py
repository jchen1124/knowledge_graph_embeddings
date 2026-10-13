
from pykeen.datasets import FB15k237
from pykeen.pipeline import pipeline
from pykeen.losses import MarginRankingLoss


# 1. Load the dataset
dataset = FB15k237()

print("Training triples:", dataset.training.num_triples)
print("Validation triples:", dataset.validation.num_triples)
print("Testing triples:", dataset.testing.num_triples)


# 2. Train the TransE model
result = pipeline(
    dataset=dataset,
    model="TransE",

    # Model settings
    model_kwargs={
        "embedding_dim": 100,
        "scoring_fct_norm": 1,
    },

    # Loss function
    loss=MarginRankingLoss,
    loss_kwargs={
        "margin": 1.0,
    },

    # Optimizer
    optimizer="Adam",
    optimizer_kwargs={
        "lr": 0.001,
    },

    # Negative sampling
    training_loop="sLCWA",
    negative_sampler="basic",
    negative_sampler_kwargs={
        "num_negs_per_pos": 1,
    },

    # Training settings
    training_kwargs={
        "num_epochs": 50,
        "batch_size": 256,
    },

    # Evaluate on validation, NOT test
    use_testing_data=False,
    evaluator_kwargs={"filtered": True},
    evaluation_kwargs={"batch_size": 32},

    # Reproducibility and hardware
    random_seed=42,
    device="cpu",
)


# 3. Save the trained model and results
result.save_to_directory("outputs/transe_50epochs")


# 4. Print validation MRR
mrr = result.metric_results.get_metric(
    "both.realistic.inverse_harmonic_mean_rank"
)

print(f"Validation Filtered MRR: {mrr:.4f}")
print("TransE training complete!")

# Validation Filtered MRR: 0.1544