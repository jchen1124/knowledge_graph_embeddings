
import csv
from pathlib import Path

import torch

from kgqa import KGQASystem, normalize_text


# 1. File paths
BASE_DIR = Path(__file__).resolve().parent
TEST_PATH = BASE_DIR / "data" / "annotated_wd_data_test.txt"

OUTPUT_DIR = BASE_DIR.parent / "report" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "q3c_predictions.csv"

TEXT_BATCH_SIZE = 32
KGE_BATCH_SIZE = 8


# 2. Load the designated test questions
records = []

with open(TEST_PATH, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")

    for head, relation, tail, question in reader:
        records.append({
            "question": question,
            "gold_head": head,
            "gold_relation": relation,
            "gold_answer": tail,
        })

print(f"Loaded {len(records)} test questions")


# 3. Initialize our existing KGQA system
system = KGQASystem()


# 4. Identify entities and relations from question text
for start in range(0, len(records), TEXT_BATCH_SIZE):
    batch = records[start:start + TEXT_BATCH_SIZE]

    questions = [
        normalize_text(row["question"])
        for row in batch
    ]

    # Entity identification using TF-IDF similarity
    question_vectors = system.entity_vectorizer.transform(
        questions
    )

    similarities = (
        system.entity_features @ question_vectors.T
    ).toarray()

    closest_indices = similarities.argmax(axis=0)

    # Relation identification using LinearSVC
    relation_vectors = system.relation_vectorizer.transform(
        questions
    )

    predicted_relations = system.relation_classifier.predict(
        relation_vectors
    )

    for j, row in enumerate(batch):
        closest = int(closest_indices[j])

        row["predicted_head"] = (
            system.training_entities[closest]
        )

        row["predicted_relation"] = predicted_relations[j]

        row["matched_training_question"] = (
            system.training_questions[closest]
        )

        row["entity_similarity"] = float(
            similarities[closest, j]
        )

print("Entity and relation identification complete!")


# 5. Prepare queries for TransE
# Includes normal predictions and oracle evaluation.
unique_pairs = []
pair_indices = {}


def register_pair(pair):
    if pair is not None and pair not in pair_indices:
        pair_indices[pair] = len(unique_pairs)
        unique_pairs.append(pair)


for row in records:

    # Predicted entity/relation
    pred_h = system.entity_to_id.get(row["predicted_head"])
    pred_r = system.relation_to_id.get(
        row["predicted_relation"]
    )

    row["pred_pair"] = (
        (pred_h, pred_r)
        if pred_h is not None and pred_r is not None
        else None
    )

    # Correct entity/relation for oracle evaluation
    gold_h = system.entity_to_id.get(row["gold_head"])
    gold_r = system.relation_to_id.get(row["gold_relation"])

    row["oracle_pair"] = (
        (gold_h, gold_r)
        if gold_h is not None and gold_r is not None
        else None
    )

    row["gold_answer_in_vocab"] = (
        row["gold_answer"] in system.entity_to_id
    )

    register_pair(row["pred_pair"])
    register_pair(row["oracle_pair"])


# 6. Score candidate answers in batches
pair_answers = {}

print(f"Scoring {len(unique_pairs)} unique queries...")

for start in range(0, len(unique_pairs), KGE_BATCH_SIZE):
    batch_pairs = unique_pairs[start:start + KGE_BATCH_SIZE]

    queries = torch.tensor(
        batch_pairs,
        dtype=torch.long,
    )

    with torch.inference_mode():
        scores = system.model.predict_t(queries)
        best_ids = scores.argmax(dim=1).cpu().tolist()

    for pair, best_id in zip(batch_pairs, best_ids):
        pair_answers[pair] = system.id_to_entity[best_id]

    if start % 256 == 0:
        print(
            f"Scored {min(start + KGE_BATCH_SIZE, len(unique_pairs))}"
            f"/{len(unique_pairs)} queries"
        )


# 7. Compare predictions against ground truth
for row in records:

    row["predicted_answer"] = pair_answers.get(
        row["pred_pair"]
    )

    row["oracle_answer"] = pair_answers.get(
        row["oracle_pair"]
    )

    row["entity_correct"] = (
        row["predicted_head"] == row["gold_head"]
    )

    row["relation_correct"] = (
        row["predicted_relation"] == row["gold_relation"]
    )

    row["answer_correct"] = (
        row["predicted_answer"] == row["gold_answer"]
    )

    row["gold_hr_in_vocab"] = (
        row["oracle_pair"] is not None
    )

    row["oracle_correct"] = (
        row["oracle_answer"] == row["gold_answer"]
        if row["gold_hr_in_vocab"]
        else False
    )

    # Error category for Question 3(d)
    if row["answer_correct"]:
        row["error_category"] = "Correct"

    elif not row["entity_correct"]:
        row["error_category"] = "Entity identification"

    elif not row["relation_correct"]:
        row["error_category"] = "Relation identification"

    elif not row["gold_answer_in_vocab"]:
        row["error_category"] = "Other: unseen answer"

    else:
        row["error_category"] = "KGE ranking"


# 8. Calculate accuracy
total = len(records)

entity_correct = sum(r["entity_correct"] for r in records)
relation_correct = sum(r["relation_correct"] for r in records)
answer_correct = sum(r["answer_correct"] for r in records)

# Oracle accuracy: only questions whose true head
# and relation are represented in the trained model.
eligible = [
    r for r in records
    if r["gold_hr_in_vocab"]
]

oracle_correct = sum(
    r["oracle_correct"] for r in eligible
)

# Additional analysis for questions whose correct
# answer also exists in the model's vocabulary.
answerable = [
    r for r in eligible
    if r["gold_answer_in_vocab"]
]

answerable_correct = sum(
    r["oracle_correct"] for r in answerable
)


def percentage(correct, count):
    return 100 * correct / count if count else 0.0


print("\n===== QUESTION 3(C): FINAL RESULTS =====")

print(
    f"Overall answer accuracy: "
    f"{answer_correct}/{total} "
    f"({percentage(answer_correct, total):.2f}%)"
)

print(
    f"Entity accuracy: "
    f"{entity_correct}/{total} "
    f"({percentage(entity_correct, total):.2f}%)"
)

print(
    f"Relation accuracy: "
    f"{relation_correct}/{total} "
    f"({percentage(relation_correct, total):.2f}%)"
)

print(
    f"Oracle KGE accuracy (known head/relation): "
    f"{oracle_correct}/{len(eligible)} "
    f"({percentage(oracle_correct, len(eligible)):.2f}%)"
)

print(
    f"Oracle KGE accuracy (all IDs known): "
    f"{answerable_correct}/{len(answerable)} "
    f"({percentage(answerable_correct, len(answerable)):.2f}%)"
)

print(
    f"Test questions with known gold head/relation: "
    f"{len(eligible)}/{total}"
)

print(
    f"Test questions with all gold IDs known: "
    f"{len(answerable)}/{total}"
)


# 9. Save individual predictions for Question 3(d)
fields = [
    "question",
    "gold_head",
    "predicted_head",
    "gold_relation",
    "predicted_relation",
    "gold_answer",
    "predicted_answer",
    "oracle_answer",
    "entity_correct",
    "relation_correct",
    "answer_correct",
    "gold_hr_in_vocab",
    "gold_answer_in_vocab",
    "oracle_correct",
    "matched_training_question",
    "entity_similarity",
    "error_category",
]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for row in records:
        writer.writerow({
            field: row.get(field)
            for field in fields
        })

print(f"\nResults saved to: {OUTPUT_PATH}")
