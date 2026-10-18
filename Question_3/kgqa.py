
import csv
import json
import re
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# 1. File locations
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data" / "annotated_wd_data_train.txt"
MODEL_DIR = BASE_DIR / "outputs" / "transe"


def normalize_text(text):
    """Normalize question text for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


class KGQASystem:

    def __init__(self):
        # 2. Load the training questions
        self.training_questions = []
        self.training_entities = []
        self.training_relations = []

        with open(TRAIN_PATH, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")

            for head, relation, tail, question in reader:
                self.training_questions.append(
                    normalize_text(question)
                )
                self.training_entities.append(head)
                self.training_relations.append(relation)

        # 3. Build the entity identification component
        # Character n-grams help match similar question wording.
        self.entity_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
        )

        self.entity_features = (
            self.entity_vectorizer.fit_transform(
                self.training_questions
            )
        )

        # 4. Train the relation identification component
        self.relation_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )

        relation_features = (
            self.relation_vectorizer.fit_transform(
                self.training_questions
            )
        )

        self.relation_classifier = LinearSVC(
            random_state=42,
            max_iter=3000,
        )

        self.relation_classifier.fit(
            relation_features,
            self.training_relations,
        )

        # 5. Load the Wikidata-to-integer mappings
        with open(MODEL_DIR / "mappings.json", "r") as file:
            mappings = json.load(file)

        self.entity_to_id = mappings["entity_to_id"]
        self.relation_to_id = mappings["relation_to_id"]

        self.id_to_entity = {
            idx: label
            for label, idx in self.entity_to_id.items()
        }

        # 6. Load the previously trained TransE model
        # Only load model files that you trust.
        self.model = torch.load(
            MODEL_DIR / "trained_model.pkl",
            map_location="cpu",
            weights_only=False,
        )

        self.model.eval()

        print("KGQA system initialized successfully!")

    def identify_entity(self, question):
        """Identify topic entity from the question text."""
        question = normalize_text(question)

        question_vector = self.entity_vectorizer.transform(
            [question]
        )

        similarities = (
            self.entity_features @ question_vector.T
        ).toarray().ravel()

        closest_index = int(similarities.argmax())

        return (
            self.training_entities[closest_index],
            self.training_questions[closest_index],
            float(similarities[closest_index]),
        )

    def identify_relation(self, question):
        """Predict the relationship expressed in the question."""
        question = normalize_text(question)

        features = self.relation_vectorizer.transform(
            [question]
        )

        return self.relation_classifier.predict(features)[0]

    def answer(self, question):
        """Run the complete KGQA pipeline."""

        # Step 1: Identify the entity
        head, matched_question, similarity = (
            self.identify_entity(question)
        )

        # Step 2: Identify the relation
        relation = self.identify_relation(question)

        # Step 3: Map identifiers to model indices
        head_id = self.entity_to_id.get(head)
        relation_id = self.relation_to_id.get(relation)

        if head_id is None or relation_id is None:
            return {
                "question": question,
                "predicted_entity": head,
                "predicted_relation": relation,
                "predicted_answer": None,
            }

        # Step 4: Construct (h, r, ?) and score all tails
        query = torch.tensor(
            [[head_id, relation_id]],
            dtype=torch.long,
        )

        with torch.no_grad():
            scores = self.model.predict_t(query).squeeze(0)

        # Step 5: Return highest-ranked entity
        predicted_tail_id = int(scores.argmax().item())

        predicted_answer = self.id_to_entity[predicted_tail_id]

        return {
            "question": question,
            "predicted_entity": head,
            "predicted_relation": relation,
            "predicted_answer": predicted_answer,
            "matched_question": matched_question,
            "entity_similarity": similarity,
        }


# 7. Interactive question-answering interface
if __name__ == "__main__":

    system = KGQASystem()

    while True:
        question = input("\nEnter a question (or 'quit'): ")

        if question.strip().lower() == "quit":
            break

        if not question.strip():
            continue

        result = system.answer(question)

        print("\n===== KGQA Result =====")
        print("Question:", result["question"])
        print("Entity:", result["predicted_entity"])
        print("Relation:", result["predicted_relation"])

        print(
            "Structured query:",
            f"({result['predicted_entity']}, "
            f"{result['predicted_relation']}, ?)"
        )

        print("Predicted answer:", result["predicted_answer"])

        if "matched_question" in result:
            print("Closest training question:",
                  result["matched_question"])
            print("Text similarity:",
                  round(result["entity_similarity"], 4))
