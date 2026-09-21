import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import preprocess_dataframe
from features import create_features
from embeddings import load_embedding_model, generate_embeddings


MODEL_PATH = "models/kmeans_model.joblib"
REFERENCE_PATH = "data/processed/video_content_analysis.csv"


def load_kmeans_model():
    """Load the existing trained K-Means model."""
    return joblib.load(MODEL_PATH)


def assign_cluster(embedding, model):
    """Assign a new video to an existing semantic cluster."""

    normalized_embedding = normalize(
        embedding,
        norm="l2"
    )

    cluster = model.predict(
        normalized_embedding
    )[0]

    return int(cluster)


def calculate_percentile(value, reference_values):
    """
    Compare a value with the reference dataset
    and return its percentile.
    """

    reference_values = np.asarray(reference_values)

    reference_values = reference_values[
        ~np.isnan(reference_values)
    ]

    if len(reference_values) == 0:
        return None

    percentile = (
        np.sum(reference_values <= value)
        / len(reference_values)
    ) * 100

    return float(percentile)


def analyze_video(video_data):
    """
    Analyze one new video.

    video_data should be a dictionary containing
    the same raw fields used by the dataset.
    """

    # Convert the video dictionary into a DataFrame
    df = pd.DataFrame([video_data])

    # Phase 1
    df = preprocess_dataframe(df)

    # Phase 2
    df = create_features(df)

    # Phase 3
    embedding_model = load_embedding_model()

    embedding = generate_embeddings(
        df,
        embedding_model
    )

    # Phase 5
    kmeans_model = load_kmeans_model()

    cluster = assign_cluster(
        embedding,
        kmeans_model
    )

    result = {
        "title": df.iloc[0].get("Title", ""),
        "cluster": cluster,
        "duration_seconds": int(
            df.iloc[0].get("duration_seconds", 0)
        ),
        "views_numeric": float(
            df.iloc[0].get("views_numeric", 0)
        ),
        "word_count": int(
            df.iloc[0].get("word_count", 0)
        ),
        "has_visual": int(
            df.iloc[0].get("has_visual", 0)
        ),
        "has_example": int(
            df.iloc[0].get("has_example", 0)
        ),
        "has_analogy": int(
            df.iloc[0].get("has_analogy", 0)
        ),
        "has_application": int(
            df.iloc[0].get("has_application", 0)
        ),
        "has_math": int(
            df.iloc[0].get("has_math", 0)
        ),
        "has_code": int(
            df.iloc[0].get("has_code", 0)
        ),
    }

    return result


if __name__ == "__main__":

    # Temporary test video
    test_video = {
        "ID": "TEST-001",
        "Category": "Machine Learning (ML)",
        "Channel": "Test Channel",
        "Title": "Introduction to Neural Networks",
        "URL": "",
        "Duration": "15:30",
        "Views": "100K+",

        "Explanation_Type":
            "Conceptual explanation with mathematical intuition",

        "Visuals":
            "Diagrams showing neurons and network layers",

        "Examples":
            "MNIST handwritten digit classification example",

        "Analogies":
            "Neurons compared with biological neurons",

        "Real_Life":
            "Image recognition and classification applications",

        "Quality": "",

        "Creator_Takeaways":
            "Understand neural networks, layers and training"
    }

    result = analyze_video(test_video)

    print("\nVideo Analysis")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")