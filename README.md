# AI/ML Video Content and Quality Analyzer

An AI/ML pipeline for analyzing educational video content using video metadata and textual descriptions.

The project extracts descriptive content indicators, generates semantic embeddings, discovers groups of similar videos using clustering, and analyzes new videos using the learned pipeline.

> **Important:** This project does not assign an overall "quality score". The reported measurements are descriptive indicators derived from the available metadata and text descriptions.

---

## Project Overview

The system analyzes educational videos using information such as:

- Video title
- Category
- Channel
- Duration
- Views
- Explanation type
- Visual descriptions
- Examples
- Analogies
- Real-life applications
- Creator takeaways

The pipeline combines traditional feature engineering with transformer-based sentence embeddings and unsupervised clustering.

---

## Dataset

The current dataset contains:

- **500 videos**
- **14 original columns**

### Original Columns

```text
ID
Category
Channel
Title
URL
Duration
Views
Explanation_Type
Visuals
Examples
Analogies
Real_Life
Quality
Creator_Takeaways
```

The project does not require manual annotation of every video.

---

# Pipeline

The project consists of the following stages:

```text
Raw Video Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Sentence Embeddings
       │
       ▼
K-Means Clustering
       │
       ▼
Cluster Evaluation
       │
       ▼
Content Analysis
       │
       ▼
New Video Prediction
```

---

# Phase 1 — Data Preprocessing

The preprocessing stage cleans the raw dataset and converts important fields into machine-readable formats.

### Main Operations

- Remove empty columns
- Clean textual fields
- Convert video duration into seconds
- Convert views such as `24.2M+` into numeric values
- Prepare cleaned data for feature engineering

### Output

```text
data/processed/videos_clean.csv
```

---

# Phase 2 — Feature Engineering

The feature engineering stage creates structured features from the cleaned dataset.

### Generated Features

```text
duration_seconds
views_numeric
log_views
has_analogy
has_example
has_visual
has_application
has_math
has_code
combined_text
text_length
word_count
```

Additional content-related indicators are generated during the analysis stage.

### Output

```text
data/processed/videos_features.csv
```

---

# Phase 3 — Semantic Embeddings

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The combined textual information from each video is converted into a semantic vector representation.

### Embedding Dimensions

```text
500 videos × 384 dimensions
```

### Generated Files

```text
data/processed/video_embeddings.npy
data/processed/video_embeddings_normalized.npy
```

---

# Phase 4 — K-Means Clustering

K-Means clustering is used to discover groups of semantically similar videos.

### Configuration

```text
Algorithm: K-Means
Number of clusters: 9
random_state: 42
n_init: 10
```

### Cluster Distribution

For the current 500-video dataset:

| Cluster | Videos |
|--------:|-------:|
| 0 | 43 |
| 1 | 49 |
| 2 | 66 |
| 3 | 62 |
| 4 | 87 |
| 5 | 39 |
| 6 | 66 |
| 7 | 37 |
| 8 | 51 |

### Model

```text
models/kmeans_model.joblib
```

---

# Phase 5 — Clustering Evaluation

The clustering results were evaluated using several metrics.

### Results

| Metric | Value |
|---|---:|
| Silhouette Score | 0.1047 |
| Adjusted Rand Index (ARI) | 0.4524 |
| Normalized Mutual Information (NMI) | 0.6454 |

These metrics describe the clustering behavior of the current dataset and configuration.

They are **not video-quality scores**.

### Metrics File

```text
results/metrics/clustering_metrics.csv
```

---

# Phase 6 — Content Analysis

The project extracts descriptive indicators from the available textual metadata.

## Explanation Description Richness

The system analyzes the amount of descriptive information associated with:

- Examples
- Analogies
- Visuals
- Practical applications
- Creator takeaways

The resulting indicator is:

```text
explanation_description_richness
```

This represents the richness of the available description fields.

It does **not** directly measure the effectiveness, correctness, or pedagogical quality of an explanation.

### Reference Statistics

For the current 500-video dataset:

```text
Mean:   0.386536
Median: 0.377720
Min:    0.140857
Max:    0.763334
```

---

## Content Coverage

Content coverage measures the presence of different content types:

```text
Visuals
Examples
Analogies
Applications
Math
Code
```

The resulting feature is:

```text
content_coverage
```

### Reference Statistics

For the current dataset:

```text
Mean:   0.774000
Median: 0.833333
Min:    0.666667
Max:    1.000000
```

---

## Technical Depth Indicator

The project also calculates:

```text
technical_depth
```

This indicator is based on the presence of technical terminology in the available textual information.

It should be interpreted as a **technical terminology indicator**, not a direct measurement of actual technical depth, correctness, or teaching effectiveness.

### Reference Statistics

For the current dataset:

```text
Mean:   0.154154
Median: 0.115385
Min:    0
Max:    1
```

---

# Phase 7 — New Video Prediction

The prediction pipeline allows a new video's information to be passed through the existing system.

The prediction process performs:

```text
New Video
    │
    ├── Preprocessing
    │
    ├── Feature Extraction
    │
    ├── Sentence Embedding
    │
    ├── K-Means Cluster Assignment
    │
    └── Content Indicator Analysis
```

The new video's description-based indicators are normalized against the existing reference dataset rather than fitting a scaler only on the single new video.

This makes the new-video indicators comparable with the reference dataset.

### Run Prediction

```powershell
python .\src\predict.py
```

---

# Example Prediction

Example test video:

```text
Title:
Introduction to Neural Networks
```

Example output:

```text
cluster: 4

duration_seconds: 930
views_numeric: 100000.0
word_count: 36

has_visual: 1
has_example: 1
has_analogy: 1
has_application: 1
has_math: 1
has_code: 0

technical_term_count: 4

explanation_description_richness: 0.03599
content_coverage: 0.83333
technical_depth: 0.28571

richness_percentile: 0.0
coverage_percentile: 91.6
technical_percentile: 72.0
```

These values are descriptive outputs from the current reference dataset and model.

---

# Project Structure

```text
project_videos/
│
├── data/
│   ├── data.csv
│   │
│   └── processed/
│       ├── videos_clean.csv
│       ├── videos_clustered.csv
│       ├── videos_features.csv
│       ├── video_content_analysis.csv
│       ├── video_embeddings.npy
│       └── video_embeddings_normalized.npy
│
├── models/
│   └── kmeans_model.joblib
│
├── results/
│   ├── figures/
│   │   ├── content_coverage_distribution.png
│   │   └── explanation_richness_distribution.png
│   │
│   └── metrics/
│       └── clustering_metrics.csv
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── embeddings.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── Data Processing.ipynb
├── embeddings.ipynb
├── feature_engineering.ipynb
├── model_training.ipynb
├── video_content_quality_analysis.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```powershell
git clone https://github.com/nagadurgarao-01/Video-Content-and-Quality-Analyzer.git
```

Navigate to the project:

```powershell
cd Video-Content-and-Quality-Analyzer
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# Running the Pipeline

## 1. Data Preprocessing

```powershell
python .\src\preprocessing.py
```

## 2. Feature Engineering

```powershell
python .\src\features.py
```

## 3. Generate Embeddings

```powershell
python .\src\embeddings.py
```

## 4. Train K-Means Model

```powershell
python .\src\train.py
```

## 5. Evaluate Clustering

```powershell
python .\src\evaluate.py
```

## 6. Analyze a New Video

```powershell
python .\src\predict.py
```

---

# Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- K-Means Clustering
- MinMaxScaler

### NLP / Embeddings

- Sentence Transformers
- `all-MiniLM-L6-v2`

### Visualization

- Matplotlib

### Model Serialization

- Joblib

### Development

- Jupyter Notebook

---

# Machine Learning Techniques

The project uses:

- Text preprocessing
- Feature engineering
- Binary content indicators
- Semantic embeddings
- Transformer-based sentence representations
- K-Means clustering
- Silhouette analysis
- Adjusted Rand Index
- Normalized Mutual Information
- Percentile-based comparison

---

# Limitations

This project analyzes the information available in the dataset rather than directly evaluating the complete video.

It currently does not analyze:

- Actual video frames
- Audio quality
- Speech delivery
- Teaching effectiveness
- Factual correctness
- Visual quality
- Instructor performance
- Student learning outcomes

Therefore, the generated indicators should be interpreted as **descriptive proxies based on metadata and textual descriptions**.

### Indicator Limitations

`technical_depth`

Measures the presence of selected technical terminology. It does not directly measure actual technical depth.

`explanation_description_richness`

Measures the richness of the available description fields. It does not measure whether an explanation is effective.

`content_coverage`

Measures the presence of selected content categories. It does not measure their quality or usefulness.

`K-Means clusters`

Represent groups of semantically similar videos. A cluster does not represent an objectively better or worse category of video.

---

# Future Work

Possible future extensions include:

- Automatic transcript extraction
- Transcript-based content analysis
- Audio and speech analysis
- Computer vision analysis of video frames
- Topic modeling
- Advanced semantic clustering
- Explainable cluster analysis
- Automatic concept extraction
- Learning-objective extraction
- Retrieval of similar educational videos
- Web-based video analysis interface
- API-based automated video analysis

---

# Reproducibility

The current clustering model uses:

```text
random_state = 42
```

The current embedding model is:

```text
all-MiniLM-L6-v2
```

The current clustering configuration is:

```text
K = 10
n_init = 10
```

The pipeline can be rerun using the scripts in the `src/` directory.

---
