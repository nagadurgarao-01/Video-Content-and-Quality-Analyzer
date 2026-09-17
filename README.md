# Video Content and Quality Analyzer 🎥📊

A data processing and feature engineering framework designed to benchmark, analyze, and quantify the educational quality and pedagogical structure of AI and Machine Learning instructional videos.

---

## 📌 Project Overview

Online AI/ML educational content varies widely in instructional pedagogy, technical depth, visual communication, and production quality. This project provides an end-to-end data pipeline to systematically analyze a benchmark dataset of **205 curated AI & ML videos** from leading creators (such as *3Blue1Brown*, *StatQuest with Josh Starmer*, and more).

The pipeline extracts qualitative attributes, normalizes metadata (durations, view counts), detects pedagogical teaching elements (visuals, analogies, real-life applications, mathematical rigor, code demonstrations), and prepares structured feature representations for downstream machine learning and embedding generation.

---

## 🗂️ Project Structure

```text
Video-Content-and-Quality-Analyzer/
│
├── data/
│   ├── Master AI & ML Video Explanation Benchmark Dataset (205 Videos) - Untitled.csv  # Raw benchmark dataset
│   └── processed/
│       ├── video_annotations.csv   # Multi-dimensional quality scoring template
│       ├── videos_clean.csv         # Cleaned and standardized video metadata
│       └── videos_features.csv      # Engineered numerical, categorical & NLP features
│
├── Data Processing.ipynb            # Phase 1: Data ingestion, cleaning & annotation setup
├── feature_engineering.ipynb        # Phase 2: Feature extraction, text synthesis & transformations
├── .gitignore                       # Git ignore rules for checkpoints and cache
└── README.md                        # Project documentation
```

---

## ⚙️ Data Pipeline Architecture

```
+-------------------------------------------------------------+
|                     Raw Benchmark Dataset                   |
| 205 AI & ML Videos (Metadata, Visuals, Quality, Takeaways)  |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|            Phase 1: Data Processing & Cleaning             |
|                  (Data Processing.ipynb)                    |
| - Prune empty/unnamed columns and duplicates                |
| - Parse Duration (MM:SS, HH:MM:SS) -> duration_seconds      |
| - Standardize Views (K, M suffixes) -> views_numeric        |
| - Construct unified combined_text representation            |
| - Generate quality annotation framework (7 dimensions)      |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|             Phase 2: Feature Engineering                    |
|               (feature_engineering.ipynb)                   |
| - Log-transform skewed view metrics (log_views)             |
| - Binary indicator extraction (Math, Code, Analogy, etc.)   |
| - Text-length and lexical density features                  |
| - Structured NLP prompt preparation for Embeddings          |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|               Engineered Benchmark Dataset                  |
|            data/processed/videos_features.csv               |
|            (Ready for ML / SentenceTransformers)            |
+-------------------------------------------------------------+
```

---

## 🔍 Key Features & Derived Metrics

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `ID` | Identifier | Unique video identifier (e.g., `VID-001`) |
| `Category` | Categorical | Subject domain (e.g., *Machine Learning*, *Deep Learning*) |
| `Channel` | Categorical | Educational YouTube channel name |
| `Title` | Text | Title of the video |
| `URL` | URL | YouTube link to the source video |
| `duration_seconds` | Numeric | Total video runtime converted to seconds |
| `views_numeric` | Numeric | View count converted to raw numerical magnitude |
| `log_views` | Numeric | Natural logarithm (`log1p`) of views to normalize distribution |
| `has_analogy` | Binary (0/1) | Whether the video incorporates conceptual analogies |
| `has_example` | Binary (0/1) | Whether concrete examples/case studies are provided |
| `has_visual` | Binary (0/1) | Whether visual demonstrations or animations are used |
| `has_application` | Binary (0/1) | Whether real-world industry applications are discussed |
| `has_math` | Binary (0/1) | Keyword detection for mathematical formulas, calculus, or derivations |
| `has_code` | Binary (0/1) | Keyword detection for code implementations (Python, PyTorch, etc.) |
| `combined_text` | Text | Consolidated pedagogical overview formatted for embedding models |
| `text_length` | Numeric | Total character length of the synthesized pedagogical description |
| `word_count` | Numeric | Total word count of the pedagogical description |

---

## 🎯 Pedagogical Quality Scoring Framework

A dedicated evaluation template (`data/processed/video_annotations.csv`) is included to evaluate instructional videos across **7 fundamental pillars**:

1. **Clarity**: Articulation and pacing of technical concepts.
2. **Correctness**: Mathematical and conceptual accuracy.
3. **Visual**: Quality and effectiveness of animations, slides, and diagrams.
4. **Examples**: Relevance and depth of illustrative problems.
5. **Analogy**: Intuitive real-world analogies bridging abstract concepts.
6. **Practical**: Relevance to practical and industry workflows.
7. **Structure**: Logical progression and modular delivery.

$$ \text{Quality Score} = \left( \frac{1}{7} \sum_{i=1}^{7} \text{Dimension}_i \right) \times 20 $$

*(Normalized to a 0–100 scale)*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nagadurgarao-01/Video-Content-and-Quality-Analyzer.git
   cd Video-Content-and-Quality-Analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install pandas numpy jupyter
   ```

### Execution

1. Open `Data Processing.ipynb` to inspect initial data ingestion, cleaning, and annotation template generation.
2. Run `feature_engineering.ipynb` to generate the complete set of engineered numerical and NLP features saved to `data/processed/videos_features.csv`.

---

## 🔮 Future Roadmap

- [ ] **SentenceTransformer Embeddings**: Vectorize `combined_text` for semantic clustering and similarity search.
- [ ] **Automated Quality Predictor**: Train regression models to predict quality scores based on video features.
- [ ] **YouTube API Integration**: Dynamically pull real-time video metrics (likes, comments, sentiment).
- [ ] **Dashboard Visualization**: Interactive Streamlit / Gradio dashboard for video exploration and ranking.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
