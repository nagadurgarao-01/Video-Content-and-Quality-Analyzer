import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize, MinMaxScaler


# ============================================================
# Allow imports from src/
# ============================================================

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import preprocess_dataframe
from features import create_features
from embeddings import load_embedding_model, generate_embeddings


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "models/kmeans_model.joblib"

REFERENCE_FEATURES_PATH = (
    "data/processed/videos_features.csv"
)


# ============================================================
# Load K-Means model
# ============================================================

def load_kmeans_model():
    """
    Load the existing trained K-Means model.
    """
    return joblib.load(MODEL_PATH)


# ============================================================
# Load reference dataset
# ============================================================

def load_reference_data():
    """
    Load the reference dataset used for Phase 6 analysis.
    """

    path = Path(REFERENCE_FEATURES_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# Cluster assignment
# ============================================================

def assign_cluster(embedding, model):
    """
    Assign a new video to an existing semantic cluster.
    """

    normalized_embedding = normalize(
        embedding,
        norm="l2"
    )

    cluster = model.predict(
        normalized_embedding
    )[0]

    return int(cluster)


# ============================================================
# Description features
# ============================================================

def calculate_description_features(
    df,
    reference_df=None
):
    """
    Calculate Phase 6 description richness.

    When reference_df is provided, MinMaxScaler is fitted
    on the reference dataset and then applied to df.

    This prevents a single new video from being normalized
    against itself.
    """

    df = df.copy()

    text_columns = {
        "example_support_raw": "Examples",
        "analogy_support_raw": "Analogies",
        "visual_support_raw": "Visuals",
        "application_support_raw": "Real_Life",
        "takeaway_richness_raw": "Creator_Takeaways",
    }

    # --------------------------------------------------------
    # Calculate raw description lengths for current dataframe
    # --------------------------------------------------------

    for new_column, source_column in text_columns.items():

        if source_column in df.columns:

            df[new_column] = (
                df[source_column]
                .fillna("")
                .astype(str)
                .str.len()
            )

        else:

            df[new_column] = 0

    raw_columns = [
        "example_support_raw",
        "analogy_support_raw",
        "visual_support_raw",
        "application_support_raw",
        "takeaway_richness_raw",
    ]

    normalized_columns = [
        "example_support",
        "analogy_support",
        "visual_support",
        "practical_support",
        "takeaway_richness",
    ]

    # --------------------------------------------------------
    # Fit scaler using reference dataset
    # --------------------------------------------------------

    if reference_df is not None:

        reference_df = reference_df.copy()

        for new_column, source_column in text_columns.items():

            if source_column in reference_df.columns:

                reference_df[new_column] = (
                    reference_df[source_column]
                    .fillna("")
                    .astype(str)
                    .str.len()
                )

            else:

                reference_df[new_column] = 0

        scaler = MinMaxScaler()

        scaler.fit(
            reference_df[raw_columns]
        )

        normalized = scaler.transform(
            df[raw_columns]
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

    # --------------------------------------------------------
    # Fallback: fit on current dataframe
    # --------------------------------------------------------

    else:

        scaler = MinMaxScaler()

        normalized = scaler.fit_transform(
            df[raw_columns]
        )

    # --------------------------------------------------------
    # Store normalized features
    # --------------------------------------------------------

    df[normalized_columns] = normalized

    # --------------------------------------------------------
    # Explanation description richness
    # --------------------------------------------------------

    df["explanation_description_richness"] = (
        df[normalized_columns]
        .mean(axis=1)
    )

    return df


# ============================================================
# Content coverage
# ============================================================

def calculate_content_coverage(df):
    """
    Calculate the Phase 6 content coverage indicator.
    """

    coverage_columns = [
        "has_visual",
        "has_example",
        "has_analogy",
        "has_application",
        "has_math",
        "has_code",
    ]

    existing_columns = [
        column
        for column in coverage_columns
        if column in df.columns
    ]

    df = df.copy()

    if len(existing_columns) == 0:

        df["content_coverage"] = 0.0

    else:

        df["content_coverage"] = (
            df[existing_columns]
            .mean(axis=1)
        )

    return df


# ============================================================
# Technical terminology
# ============================================================

def count_technical_terms(text):
    """
    Count technical terminology in the combined text.

    This is a terminology indicator, not a measure
    of actual conceptual depth.
    """

    if pd.isna(text):
        return 0

    text = str(text).lower()

    technical_terms = [
        "neural network",
        "machine learning",
        "deep learning",
        "transformer",
        "attention",
        "self-attention",
        "embedding",
        "vector",
        "gradient descent",
        "backpropagation",
        "classification",
        "regression",
        "clustering",
        "reinforcement learning",
        "convolution",
        "cnn",
        "rnn",
        "lstm",
        "bert",
        "gpt",
        "llm",
        "rag",
        "retrieval",
        "fine-tuning",
        "fine tuning",
        "lora",
        "diffusion",
        "generative ai",
        "computer vision",
        "nlp",
        "natural language processing",
        "tokenization",
        "token",
        "pytorch",
        "tensorflow",
        "yolo",
        "gan",
        "vae",
        "q-learning",
        "deep q-learning",
        "reinforcement",
        "agent",
        "multi-agent",
        "vector database",
        "reranking",
        "knowledge graph",
    ]

    count = 0

    for term in technical_terms:
        count += text.count(term)

    return count


# ============================================================
# Technical depth
# ============================================================

def calculate_technical_depth(
    df,
    reference_df=None
):
    """
    Calculate the Phase 6 technical terminology indicator.

    When reference_df is provided, normalization is based
    on the reference dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Count technical terms in current dataframe
    # --------------------------------------------------------

    df["technical_term_count"] = (
        df["combined_text"]
        .fillna("")
        .apply(count_technical_terms)
    )

    # --------------------------------------------------------
    # Fit scaler using reference dataset
    # --------------------------------------------------------

    if reference_df is not None:

        reference_df = reference_df.copy()

        reference_df["technical_term_count"] = (
            reference_df["combined_text"]
            .fillna("")
            .apply(count_technical_terms)
        )

        scaler = MinMaxScaler()

        scaler.fit(
            reference_df[
                ["technical_term_count"]
            ]
        )

        df["technical_depth"] = (
            scaler.transform(
                df[
                    ["technical_term_count"]
                ]
            )
            .flatten()
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    else:

        scaler = MinMaxScaler()

        df["technical_depth"] = (
            scaler.fit_transform(
                df[
                    ["technical_term_count"]
                ]
            )
            .flatten()
        )

    return df


# ============================================================
# Phase 6 indicators
# ============================================================

def calculate_phase6_indicators(
    df,
    reference_df=None
):
    """
    Calculate all Phase 6 indicators.
    """

    df = df.copy()

    # Explanation richness
    df = calculate_description_features(
        df,
        reference_df
    )

    # Content coverage
    df = calculate_content_coverage(
        df
    )

    # Technical depth
    df = calculate_technical_depth(
        df,
        reference_df
    )

    return df


# ============================================================
# Percentile calculation
# ============================================================

def calculate_percentile(
    value,
    reference_values
):
    """
    Calculate percentile relative to the reference dataset.
    """

    reference_values = np.asarray(
        reference_values,
        dtype=float
    )

    reference_values = reference_values[
        ~np.isnan(reference_values)
    ]

    if len(reference_values) == 0:
        return None

    return float(
        np.sum(reference_values <= value)
        / len(reference_values)
        * 100
    )


# ============================================================
# Analyze one video
# ============================================================

def analyze_video(video_data):
    """
    Analyze one new video using the existing
    500-video reference dataset.
    """

    # ========================================================
    # Phase 1: Preprocessing
    # ========================================================

    df = pd.DataFrame(
        [video_data]
    )

    df = preprocess_dataframe(
        df
    )

    # ========================================================
    # Phase 2: Feature engineering
    # ========================================================

    df = create_features(
        df
    )

    # ========================================================
    # Phase 3: Embedding
    # ========================================================

    embedding_model = load_embedding_model()

    embedding = generate_embeddings(
        df,
        embedding_model
    )

    # ========================================================
    # Phase 5: Cluster prediction
    # ========================================================

    kmeans_model = load_kmeans_model()

    cluster = assign_cluster(
        embedding,
        kmeans_model
    )

    # ========================================================
    # Phase 6: Load reference dataset
    # ========================================================

    reference_df = load_reference_data()

    # Calculate indicators for the reference dataset.
    #
    # Since reference_df is being passed to itself,
    # the scalers are fitted using the complete reference
    # distribution.

    reference_analysis = (
        calculate_phase6_indicators(
            reference_df
        )
    )

    # ========================================================
    # Phase 6: Calculate indicators for new video
    #
    # IMPORTANT:
    # The reference dataset is supplied here so that the
    # new video's values are transformed using the same
    # reference distribution.
    # ========================================================

    new_video_analysis = (
        calculate_phase6_indicators(
            df,
            reference_df
        )
    )

    new_video = (
        new_video_analysis
        .iloc[0]
    )

    # ========================================================
    # Extract indicators
    # ========================================================

    richness = float(
        new_video[
            "explanation_description_richness"
        ]
    )

    coverage = float(
        new_video[
            "content_coverage"
        ]
    )

    technical_depth = float(
        new_video[
            "technical_depth"
        ]
    )

    # ========================================================
    # Percentiles
    # ========================================================

    richness_percentile = (
        calculate_percentile(
            richness,
            reference_analysis[
                "explanation_description_richness"
            ].values
        )
    )

    coverage_percentile = (
        calculate_percentile(
            coverage,
            reference_analysis[
                "content_coverage"
            ].values
        )
    )

    technical_percentile = (
        calculate_percentile(
            technical_depth,
            reference_analysis[
                "technical_depth"
            ].values
        )
    )

    # ========================================================
    # Final result
    # ========================================================

    result = {

        "title": new_video.get(
            "Title",
            ""
        ),

        "cluster": cluster,

        "duration_seconds": int(
            new_video.get(
                "duration_seconds",
                0
            )
        ),

        "views_numeric": float(
            new_video.get(
                "views_numeric",
                0
            )
        ),

        "word_count": int(
            new_video.get(
                "word_count",
                0
            )
        ),

        "has_visual": int(
            new_video.get(
                "has_visual",
                0
            )
        ),

        "has_example": int(
            new_video.get(
                "has_example",
                0
            )
        ),

        "has_analogy": int(
            new_video.get(
                "has_analogy",
                0
            )
        ),

        "has_application": int(
            new_video.get(
                "has_application",
                0
            )
        ),

        "has_math": int(
            new_video.get(
                "has_math",
                0
            )
        ),

        "has_code": int(
            new_video.get(
                "has_code",
                0
            )
        ),

        "technical_term_count": int(
            new_video.get(
                "technical_term_count",
                0
            )
        ),

        "explanation_description_richness": (
            richness
        ),

        "content_coverage": (
            coverage
        ),

        "technical_depth": (
            technical_depth
        ),

        "richness_percentile": (
            richness_percentile
        ),

        "coverage_percentile": (
            coverage_percentile
        ),

        "technical_percentile": (
            technical_percentile
        ),
    }

    return result


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test video
    # --------------------------------------------------------

    test_video = {

        "ID": "TEST-001",

        "Category":
            "Machine Learning (ML)",

        "Channel":
            "Test Channel",

        "Title":
            "Introduction to Neural Networks",

        "URL":
            "",

        "Duration":
            "15:30",

        "Views":
            "100K+",

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

        "Quality":
            "",

        "Creator_Takeaways":
            "Understand neural networks, layers and training"
    }

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    result = analyze_video(
        test_video
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("Video Analysis")
    print("=" * 50)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )