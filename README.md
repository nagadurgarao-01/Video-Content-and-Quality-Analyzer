# Video Content and Quality Analyzer 🎥📊

A data processing and feature engineering framework designed to benchmark, analyze, and quantify the educational quality and pedagogical structure of AI and Machine Learning instructional videos.

---

## 📌 Project Overview

Online AI/ML educational content varies widely in instructional pedagogy, technical depth, visual communication, and production quality. This project provides an end-to-end data pipeline to systematically analyze a benchmark dataset of **205 curated AI & ML videos** from leading creators (such as *3Blue1Brown*, *StatQuest with Josh Starmer*, and more).

The pipeline extracts qualitative attributes, normalizes metadata (durations, view counts), detects pedagogical teaching elements (visuals, analogies, real-life applications, mathematical rigor, code demonstrations), and prepares structured feature representations for downstream machine learning and embedding generation.

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


