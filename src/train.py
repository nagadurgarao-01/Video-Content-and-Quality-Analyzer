import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


N_CLUSTERS = 9
RANDOM_STATE = 42
N_INIT = 10


def load_embeddings(path):
    """Load embeddings from a NumPy file."""
    return np.load(path)


def normalize_embeddings(embeddings):
    """L2-normalize embeddings."""
    return normalize(embeddings, norm="l2")


def train_kmeans(embeddings):
    """Train the K-Means clustering model."""

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    model.fit(embeddings)

    return model


def save_model(model, path):
    """Save the trained clustering model."""
    joblib.dump(model, path)


def load_model(path):
    """Load a previously trained clustering model."""
    return joblib.load(path)


def create_clustered_dataset(
    model,
    embeddings,
    input_csv,
    output_csv
):
    """Add cluster labels to the video dataset."""

    df = pd.read_csv(input_csv)

    cluster_labels = model.predict(embeddings)

    if len(df) != len(cluster_labels):
        raise ValueError(
            "Number of videos and embeddings do not match."
        )

    df["cluster"] = cluster_labels

    df.to_csv(
        output_csv,
        index=False
    )

    return df


if __name__ == "__main__":

    embeddings_path = (
        "data/processed/video_embeddings.npy"
    )

    features_path = (
        "data/processed/videos_features.csv"
    )

    model_path = (
        "models/kmeans_model.joblib"
    )

    clustered_path = (
        "data/processed/videos_clustered.csv"
    )

    # Load embeddings
    embeddings = load_embeddings(
        embeddings_path
    )

    # Normalize embeddings
    normalized_embeddings = normalize_embeddings(
        embeddings
    )

    # Train K-Means
    model = train_kmeans(
        normalized_embeddings
    )

    # Save model
    save_model(
        model,
        model_path
    )

    # Create clustered dataset
    clustered_df = create_clustered_dataset(
        model,
        normalized_embeddings,
        features_path,
        clustered_path
    )

    print("K-Means training completed.")
    print("Clusters:", N_CLUSTERS)
    print("Number of samples:", len(embeddings))

    print(
        "Model saved to:",
        model_path
    )

    print(
        "Clustered dataset saved to:",
        clustered_path
    )

    print("\nCluster distribution:")
    print(
        clustered_df["cluster"]
        .value_counts()
        .sort_index()
    )