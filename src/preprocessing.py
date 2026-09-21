import pandas as pd
import re


TEXT_COLUMNS = [
    "Category",
    "Channel",
    "Title",
    "URL",
    "Explanation_Type",
    "Visuals",
    "Examples",
    "Analogies",
    "Real_Life",
    "Quality",
    "Creator_Takeaways",
]


def clean_text(value):
    """Clean a text value while preserving its meaning."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def duration_to_seconds(value):
    """Convert MM:SS or HH:MM:SS duration into seconds."""
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if not value:
        return 0

    parts = value.split(":")

    try:
        parts = [int(part) for part in parts]

        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds

    except ValueError:
        return 0

    return 0


def views_to_numeric(value):
    """Convert view counts such as 24.2M+ or 950K+ into numbers."""
    if pd.isna(value):
        return 0

    value = str(value).strip().upper()

    if not value:
        return 0

    value = value.replace(",", "").replace("+", "")

    multiplier = 1

    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]

    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]

    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value[:-1]

    try:
        return float(value) * multiplier
    except ValueError:
        return 0


def preprocess_dataframe(df):
    """
    Preprocess raw video dataset.

    Returns:
        pd.DataFrame: cleaned dataframe
    """

    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean text columns
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    # Convert duration
    if "Duration" in df.columns:
        df["duration_seconds"] = df["Duration"].apply(
            duration_to_seconds
        )

    # Convert views
    if "Views" in df.columns:
        df["views_numeric"] = df["Views"].apply(
            views_to_numeric
        )

    return df


def preprocess_csv(input_path, output_path=None):
    """
    Load a CSV, preprocess it, and optionally save the result.
    """

    df = pd.read_csv(input_path)

    df = preprocess_dataframe(df)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":

    input_path = "data/data.csv"
    output_path = "data/processed/videos_clean.csv"

    df = preprocess_csv(
        input_path,
        output_path
    )

    print("Preprocessing completed.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())