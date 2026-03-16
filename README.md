# AI News Intelligence Platform

An automated pipeline for collecting, analyzing, and delivering high-quality news intelligence.

## Overview

The AI News Intelligence Platform is designed to transform the noise of global news into actionable signals. It leverages state-of-the-art LLMs (like Mistral-7B) to perform technical summarization, sentiment analysis, and entity extraction.

## Current Progress

- ✅ **Phase 1: Ingestion Layer**: Robust collection from NewsAPI and custom RSS feeds (BBC, Reuters, etc.).
- ✅ **Phase 2: Data Preprocessing**: HTML cleaning, unicode normalization, and Hugging Face Dataset integration.
- ✅ **Phase 3: Intelligence Layer**: Inference engine with structured prompt templates for deep news analysis.
- 🚧 **Phase 4: Distribution (In Progress)**: Telegram Bot integration for real-time delivery of intelligence reports.

## Architecture

1.  **Ingestion**: `src/ingestion/` - Fetches articles and saves daily raw JSON.
2.  **Processing**: `src/data/` - Cleans raw data and manages the `NewsDataset`.
3.  **Inference**: `src/inference/` - Orchestrates LLM analysis (Mistral-7B).
4.  **Utility**: `src/utils/` - Centralized config and logging.

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- A GPU with 12GB+ VRAM (for full-scale inference) or use `--tiny` for testing.

### Installation
```bash
poetry install
```

### Usage

1. **Collect News**:
   ```bash
   python scripts/collect_news.py
   ```

2. **Preprocess Data**:
   ```bash
   python scripts/preprocess_data.py
   ```

3. **Analyze Intelligence**:
   ```bash
   python scripts/analyze_news.py --limit 5
   ```

## License
MIT