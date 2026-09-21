import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def load_embeddings(path):
    """Load normalized embeddings."""
    return np.load(path)


def load_model(path):
    """Load the trained K-Means model."""
    return joblib.load(path)


def load_categories(path):
    """Load the known category labels."""
    df = pd.read_csv(path)

    if "Category" not in df.columns:
        raise ValueError("Category column not found.")

    return df["Category"].values


def evaluate_clustering(
    embeddings,
    cluster_labels,
    true_labels
):
    """Calculate clustering evaluation metrics."""

    silhouette = silhouette_score(
        embeddings,
        cluster_labels
    )

    ari = adjusted_rand_score(
        true_labels,
        cluster_labels
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        cluster_labels
    )

    return {
        "silhouette_score": silhouette,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
    }


def save_metrics(metrics, output_path):
    """Save evaluation metrics to a CSV file."""

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    embeddings_path = (
        "data/processed/"
        "video_embeddings_normalized.npy"
    )

    model_path = "models/kmeans_model.joblib"

    data_path = (
        "data/processed/"
        "videos_clustered.csv"
    )

    output_path = (
        "results/metrics/"
        "clustering_metrics.csv"
    )

    embeddings = load_embeddings(
        embeddings_path
    )

    model = load_model(
        model_path
    )

    df = pd.read_csv(data_path)

    if "Category" not in df.columns:
        raise ValueError(
            "Category column not found in videos_clustered.csv."
        )

    true_labels = df["Category"].values

    cluster_labels = model.predict(
        embeddings
    )

    metrics = evaluate_clustering(
        embeddings,
        cluster_labels,
        true_labels
    )

    save_metrics(
        metrics,
        output_path
    )

    print("Clustering evaluation completed.")
    print()
    print(
        f"Silhouette Score: "
        f"{metrics['silhouette_score']:.4f}"
    )

    print(
        f"Adjusted Rand Index: "
        f"{metrics['adjusted_rand_index']:.4f}"
    )

    print(
        f"Normalized Mutual Information: "
        f"{metrics['normalized_mutual_information']:.4f}"
    )

    print()
    print(
        f"Metrics saved to: {output_path}"
    )