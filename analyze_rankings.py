
from collections import Counter, defaultdict
from pathlib import Path
import csv
import random

import torch
from pykeen.datasets import FB15k237


# 1. Load dataset
dataset = FB15k237()

training = dataset.training
validation = dataset.validation
testing = dataset.testing

# Paths to previously trained models
MODEL_PATHS = {
    "TransE": "outputs/transe_50epochs/trained_model.pkl",
    "RotatE": "outputs/rotate_50epochs/trained_model.pkl",
    "RESCAL": "outputs/rescal_reg001_10epochs/trained_model.pkl",
}

OUTPUT_DIR = Path("report/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert PyKEEN integer IDs into Freebase labels
entity_labels = {
    idx: label
    for label, idx in training.entity_to_id.items()
}

relation_labels = {
    idx: label
    for label, idx in training.relation_to_id.items()
}


# 2. Select three test triples with different relations
rng = random.Random(42)
test_triples = testing.mapped_triples.tolist()

preferred_relations = [
    "/film/film/genre",
    "/film/film/directed_by",
    "/people/person/nationality",
]

# Group test triples by relation
by_relation = defaultdict(list)

for h, r, t in test_triples:
    by_relation[r].append((h, r, t))

selected_relations = []

for label in preferred_relations:
    relation_id = training.relation_to_id.get(label)

    if (
        relation_id is not None
        and relation_id in by_relation
        and relation_id not in selected_relations
    ):
        selected_relations.append(relation_id)

# Fallback: use the most frequent remaining relations
counts = Counter(r for _, r, _ in test_triples)

for relation_id, _ in counts.most_common():
    if len(selected_relations) == 3:
        break

    if relation_id not in selected_relations:
        selected_relations.append(relation_id)

selected_triples = [
    rng.choice(by_relation[r])
    for r in selected_relations
]


# 3. Collect known true tails for filtered ranking
known_true_tails = defaultdict(set)

for split in [training, validation, testing]:
    for h, r, t in split.mapped_triples.tolist():
        known_true_tails[(h, r)].add(t)


# 4. Load trained models
models = {}

for name, path in MODEL_PATHS.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing model: {path}")

    # Only load models you trained and trust
    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()
    models[name] = model


# 5. Score and rank predictions
def get_predictions(model, head, relation, true_tail):

    # Construct (h, r, ?) query
    query = torch.tensor(
        [[head, relation]],
        dtype=torch.long,
    )

    with torch.no_grad():
        scores = model.predict_t(query).squeeze(0).cpu().clone()

    # Remove other known correct answers.
    # Keep the target answer for ranking.
    for tail in known_true_tails[(head, relation)]:
        if tail != true_tail:
            scores[tail] = float("-inf")

    # Calculate true-answer rank (realistic tie handling)
    true_score = scores[true_tail]
    higher = (scores > true_score).sum().item()
    equal = (scores == true_score).sum().item()

    true_rank = 1 + higher + (equal - 1) / 2

    # Retrieve five highest-ranked entities
    top_five = torch.topk(scores, k=5).indices.tolist()

    return true_rank, top_five


# 6. Analyze each selected triple
rows = []

for query_num, (h, r, t) in enumerate(selected_triples, start=1):

    print(f"\n{'=' * 65}")
    print(f"QUERY {query_num}")
    print(f"Head: {entity_labels[h]}")
    print(f"Relation: {relation_labels[r]}")
    print(f"True answer: {entity_labels[t]}")
    print(f"{'=' * 65}")

    for model_name, model in models.items():

        true_rank, predictions = get_predictions(
            model, h, r, t
        )

        print(f"\n{model_name}")
        print(f"True answer rank: {true_rank:g}")
        print("Top 5 predictions:")

        for rank, entity_id in enumerate(predictions, start=1):

            label = entity_labels[entity_id]
            correct = entity_id == t

            marker = " <-- TRUE ANSWER" if correct else ""

            print(f"  {rank}. {label}{marker}")

            rows.append({
                "query": query_num,
                "head": entity_labels[h],
                "relation": relation_labels[r],
                "true_answer": entity_labels[t],
                "model": model_name,
                "true_answer_rank": true_rank,
                "prediction_rank": rank,
                "predicted_entity": label,
                "is_true_answer": correct,
            })


# 7. Export results
output_path = OUTPUT_DIR / "q1e_rankings.csv"

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "query",
            "head",
            "relation",
            "true_answer",
            "model",
            "true_answer_rank",
            "prediction_rank",
            "predicted_entity",
            "is_true_answer",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved to {output_path}")
