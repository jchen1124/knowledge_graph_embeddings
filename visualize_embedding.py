
from collections import defaultdict
from pathlib import Path
import csv
import random
import matplotlib.pyplot as plt
import torch
from pykeen.datasets import FB15k237
from sklearn.decomposition import PCA


# 1. Locations of your trained models
MODEL_PATHS = {
    "TransE": "outputs/transe_50epochs/trained_model.pkl",
    "RotatE": "outputs/rotate_50epochs/trained_model.pkl",
    "RESCAL": "outputs/rescal_reg001_10epochs/trained_model.pkl",
}

# Save final figures outside the ignored outputs/ directory
OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible entity selection
rng = random.Random(42)

# 2. Load the same dataset used for training
dataset = FB15k237()
training = dataset.training

# Convert entity IDs back to Freebase labels
id_to_label = {
    entity_id: label
    for label, entity_id in training.entity_to_id.items()
}

# 3. Select a meaningful subset based on film genres
relation = "/film/film/genre"
relation_id = training.relation_to_id[relation]

# Group movies by their genre
genre_to_movies = defaultdict(set)

for head, rel, tail in training.mapped_triples.tolist():
    if rel == relation_id:
        genre_to_movies[tail].add(head)

# Select 3 genres with 12 distinct movies each
groups = []
used_movies = set()

for genre_id, movies in sorted(
    genre_to_movies.items(),
    key=lambda item: (-len(item[1]), item[0]),
):
    available = sorted(movies - used_movies)

    if len(available) < 12:
        continue

    selected = rng.sample(available, 12)
    groups.append((genre_id, selected))
    used_movies.update(selected)

    if len(groups) == 3:
        break

if len(groups) < 3:
    raise ValueError("Could not find three suitable genre groups.")

# Flatten the selected movie IDs
entity_ids = [
    movie_id
    for _, movies in groups
    for movie_id in movies
]

# Record the selected entities for reproducibility
with (OUTPUT_DIR / "selected_movies.csv").open(
    "w", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["movie_id", "genre_id"])

    for genre_id, movies in groups:
        for movie_id in movies:
            writer.writerow([
                id_to_label[movie_id],
                id_to_label[genre_id],
            ])


# 4. Load each trained model and create PCA plots
for model_name, model_path in MODEL_PATHS.items():

    print(f"Visualizing {model_name}...")

    # Only load model files that you trained and trust
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    # Extract embeddings for the same selected movies
    indices = torch.tensor(entity_ids, dtype=torch.long)

    with torch.no_grad():
        embeddings = model.entity_representations[0](
            indices=indices
        ).detach().cpu()

    # RotatE uses complex-valued embeddings.
    # Preserve both real and imaginary components.
    if torch.is_complex(embeddings):
        embeddings = torch.view_as_real(embeddings)

    embeddings = embeddings.reshape(len(entity_ids), -1)
    embeddings = embeddings.numpy()

    # 5. Project embeddings into 2D using PCA
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(embeddings)

    # 6. Plot each genre group
    fig, ax = plt.subplots(figsize=(10, 7))

    for group_index, (genre_id, movies) in enumerate(groups):
        start = group_index * 12
        end = start + 12

        points = coordinates[start:end]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=f"Genre {id_to_label[genre_id]}",
            s=65,
            alpha=0.8,
        )

        # Label a few entities without overcrowding the plot
        for i in range(3):
            ax.annotate(
                id_to_label[movies[i]],
                (points[i, 0], points[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(f"{model_name} Entity Embeddings (PCA)")

    ax.set_xlabel(
        f"Principal Component 1 "
        f"({pca.explained_variance_ratio_[0]:.1%} variance)"
    )
    ax.set_ylabel(
        f"Principal Component 2 "
        f"({pca.explained_variance_ratio_[1]:.1%} variance)"
    )

    ax.legend(title="Shared film genre")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    # Save the figure for the report
    output_path = OUTPUT_DIR / f"{model_name.lower()}_pca.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")

print("All PCA visualizations complete!")
