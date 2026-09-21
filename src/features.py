import pandas as pd
import numpy as np
import re


def create_combined_text(df):
    """Combine the main explanatory text fields."""

    text_columns = [
        "Title",
        "Explanation_Type",
        "Visuals",
        "Examples",
        "Analogies",
        "Real_Life",
        "Creator_Takeaways",
    ]

    existing_columns = [
        column for column in text_columns
        if column in df.columns
    ]

    df["combined_text"] = (
        df[existing_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    return df


def create_content_indicators(df):
    """Create binary indicators for explanation content."""

    df["has_analogy"] = (
        df["Analogies"].fillna("").str.strip().str.len() > 0
    ).astype(int)

    df["has_example"] = (
        df["Examples"].fillna("").str.strip().str.len() > 0
    ).astype(int)

    df["has_visual"] = (
        df["Visuals"].fillna("").str.strip().str.len() > 0
    ).astype(int)

    df["has_application"] = (
        df["Real_Life"].fillna("").str.strip().str.len() > 0
    ).astype(int)

    # Detect mathematical notation / mathematical terms
    math_pattern = (
        r"\b(math|mathematical|equation|formula|"
        r"calculus|probability|statistics|matrix|vector|"
        r"derivative|integral|algebra|geometry)\b"
    )

    df["has_math"] = (
        df["combined_text"]
        .str.contains(math_pattern, case=False, regex=True, na=False)
    ).astype(int)

    # Detect programming/code-related content
    code_pattern = (
        r"\b(code|coding|programming|python|javascript|"
        r"java|implementation|github|pytorch|tensorflow|"
        r"implementation)\b"
    )

    df["has_code"] = (
        df["combined_text"]
        .str.contains(code_pattern, case=False, regex=True, na=False)
    ).astype(int)

    return df


def create_text_statistics(df):
    """Create basic text statistics."""

    df["text_length"] = df["combined_text"].str.len()

    df["word_count"] = (
        df["combined_text"]
        .str.split()
        .str.len()
    )

    return df


def create_log_views(df):
    """Create logarithmic view-count feature."""

    if "views_numeric" in df.columns:
        df["log_views"] = np.log1p(df["views_numeric"])
    elif "Views" in df.columns:
        raise ValueError(
            "views_numeric is missing. "
            "Run preprocessing.py first."
        )

    return df


def create_features(df):
    """
    Apply the complete Phase 2 feature-engineering pipeline.

    Input:
        Preprocessed video dataframe

    Output:
        Dataframe containing engineered features
    """

    df = df.copy()

    df = create_combined_text(df)

    df = create_content_indicators(df)

    df = create_text_statistics(df)

    df = create_log_views(df)

    return df


def create_features_from_csv(input_path, output_path=None):
    """
    Load a preprocessed CSV, create features,
    and optionally save the result.
    """

    df = pd.read_csv(input_path)

    df = create_features(df)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":

    input_path = "data/processed/videos_clean.csv"
    output_path = "data/processed/videos_features.csv"

    df = create_features_from_csv(
        input_path,
        output_path
    )

    print("Feature engineering completed.")
    print("Shape:", df.shape)

    print("\nNew features:")
    print([
        "combined_text",
        "has_analogy",
        "has_example",
        "has_visual",
        "has_math",
        "has_code",
        "has_application",
        "log_views",
        "text_length",
        "word_count",
    ])

    print("\nFeature columns:")
    print(df.columns.tolist())