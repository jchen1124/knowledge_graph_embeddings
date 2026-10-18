
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from pykeen.losses import MarginRankingLoss
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory


# 1. Set up paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "annotated_wd_data_train.txt"

OUTPUT_DIR = BASE_DIR / "outputs" / "transe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# 3. Load unique training triples
triples = set()

with open(DATA_PATH, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")

    for subject, relation, object_, question in reader:
        triples.add((subject, relation, object_))

# PyKEEN expects an array with 3 columns: h, r, t
triples_array = np.asarray(
    sorted(triples),
    dtype=str,
)


# 4. Convert triples into PyKEEN format
training = TriplesFactory.from_labeled_triples(
    triples_array,
    create_inverse_triples=False,
)

print("Training triples:", training.num_triples)
print("Entities:", training.num_entities)
print("Relations:", training.num_relations)


# 5. Initialize TransE
model = TransE(
    triples_factory=training,
    embedding_dim=100,
    scoring_fct_norm=1,
    loss=MarginRankingLoss(margin=1.0),
)


# 6. Configure optimizer and training loop
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training,
    optimizer=optimizer,
    negative_sampler="basic",
    negative_sampler_kwargs={
        "num_negs_per_pos": 1,
    },
)


# 7. Train the model (initial test run)
losses = training_loop.train(
    triples_factory=training,
    num_epochs=50,
    batch_size=256,
)

print("Final training loss:", losses[-1])


# 8. Save model and mappings
torch.save(
    model,
    OUTPUT_DIR / "trained_model.pkl",
)

# Preserve the exact entity/relation ID mappings for KGQA
with open(OUTPUT_DIR / "mappings.json", "w") as file:
    json.dump(
        {
            "entity_to_id": dict(training.entity_to_id),
            "relation_to_id": dict(training.relation_to_id),
        },
        file,
        indent=2,
    )

print("Training complete!")
print("Model saved to:", OUTPUT_DIR)
