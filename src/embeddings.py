import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(df, model=None):
    """
    Generate embeddings from combined_text.

    Returns:
        numpy.ndarray of shape (number_of_videos, 384)
    """

    if "combined_text" not in df.columns:
        raise ValueError(
            "combined_text column is missing. "
            "Run feature engineering first."
        )

    if model is None:
        model = load_embedding_model()

    texts = df["combined_text"].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


def save_embeddings(embeddings, output_path):
    """Save embeddings as a NumPy file."""
    np.save(output_path, embeddings)


def load_embeddings(input_path):
    """Load previously saved embeddings."""
    return np.load(input_path)


def generate_embeddings_from_csv(
    input_path,
    output_path
):
    """
    Load feature-engineered data,
    generate embeddings, and save them.
    """

    df = pd.read_csv(input_path)

    model = load_embedding_model()

    embeddings = generate_embeddings(
        df,
        model
    )

    save_embeddings(
        embeddings,
        output_path
    )

    return embeddings


if __name__ == "__main__":

    input_path = "data/processed/videos_features.csv"
    output_path = "data/processed/video_embeddings.npy"

    embeddings = generate_embeddings_from_csv(
        input_path,
        output_path
    )

    print("Embedding generation completed.")
    print("Embedding shape:", embeddings.shape)