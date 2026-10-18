
import csv
from pathlib import Path
from rdflib import Graph, URIRef, Namespace


# 1. File locations
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "annotated_wd_data_train.txt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. Define namespaces for entities and relations
ENTITY = Namespace("http://www.wikidata.org/entity/")
RELATION = Namespace("https://example.org/kgqa/relation/")


# 3. Create the RDF knowledge graph
graph = Graph()

sample_triples = []
seen_relations = set()

with open(DATA_PATH, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")

    for subject, relation, object_, question in reader:

        # Convert identifiers to RDF URIs
        head = ENTITY[subject]
        predicate = RELATION[relation]
        tail = ENTITY[object_]

        # Insert the triple into the graph
        graph.add((head, predicate, tail))

        # Save two examples with different relations
        if relation not in seen_relations and len(sample_triples) < 2:
            sample_triples.append(
                (subject, relation, object_, question)
            )
            seen_relations.add(relation)


# 4. Calculate basic graph statistics
entities = set(graph.subjects()) | set(graph.objects())
relations = set(graph.predicates())

print("===== Knowledge Graph Statistics =====")
print(f"Entities: {len(entities)}")
print(f"Relations: {len(relations)}")
print(f"Triples: {len(graph)}")


# 5. Demonstrate two SPARQL queries
for i, (subject, relation, object_, question) in enumerate(
    sample_triples, start=1
):
    print(f"\n===== Query {i} =====")
    print(f"Question: {question}")
    print(f"Expected answer ID: {object_}")

    # Retrieve all connected tail entities
    sparql = f"""
        SELECT ?answer
        WHERE {{
            <{ENTITY[subject]}> <{RELATION[relation]}> ?answer .
        }}
    """

    results = graph.query(sparql)

    for row in results:
        answer_id = str(row.answer).split("/")[-1]
        print(f"Retrieved answer: {answer_id}")


# 6. Save the resulting graph
output_path = OUTPUT_DIR / "training_graph.ttl"

graph.serialize(
    destination=str(output_path),
    format="turtle",
)

print(f"\nGraph saved to: {output_path}")
