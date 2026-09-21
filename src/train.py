import numpy as np
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


if __name__ == "__main__":

    input_path = "data/processed/video_embeddings.npy"
    model_path = "models/kmeans_model.joblib"

    embeddings = load_embeddings(input_path)

    normalized_embeddings = normalize_embeddings(
        embeddings
    )

    model = train_kmeans(
        normalized_embeddings
    )

    save_model(
        model,
        model_path
    )

    print("K-Means training completed.")
    print("Clusters:", N_CLUSTERS)
    print("Number of samples:", len(embeddings))
    print("Model saved to:", model_path)