# AI News Intelligence Platform

An automated pipeline for collecting, analyzing, and delivering high-quality news intelligence.

## Overview

The AI News Intelligence Platform is designed to transform the noise of global news into actionable signals. It leverages state-of-the-art LLMs (like Mistral-7B) to perform technical summarization, sentiment analysis, and entity extraction.

## Current Progress

- ✅ **Phase 1: Ingestion Layer**: Robust collection from NewsAPI and custom RSS feeds.
- ✅ **Phase 2: Data Preprocessing**: HTML cleaning and Hugging Face Dataset integration.
- ✅ **Phase 3: Intelligence Layer**: Inference engine with structured prompt templates.
- 🧪 **Phase 4: Model Experiments**: Tiny-model and training scaffolding exists; model training is experimental and not part of the daily pipeline.
- ✅ **Phase 5: Aggregation & Reporting**: Basic aggregation and Markdown report generation.
- ✅ **Phase 6: Telegram Delivery**: Message and file delivery to a configured Telegram chat.
- ✅ **Phase 7: Pipeline Orchestration**: Single-command local execution. GitHub scheduling and the Kali runner are still planned.

## Architecture

1.  **Ingestion**: `src/ingestion/` - Fetches articles and saves daily raw JSON.
2.  **Processing**: `src/data/` - Cleans raw data and manages the `NewsDataset`.
3.  **Inference**: `src/inference/` - Orchestrates LLM analysis (Mistral-7B).
4.  **Model Experiments**: `src/training/` - Optional training experiments, separate from the daily inference path.
5.  **Reporting**: `src/reporting/` - Aggregates article analyses into themes and formats reports.
6.  **Delivery**: `src/delivery/` - Sends intelligence reports via Telegram.
7.  **Utility**: `src/utils/` - Centralized config and logging.

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- A GPU with 12GB+ VRAM (for full-scale inference) or use `--tiny` for testing.

The current inference configuration uses Mistral. The planned local Bonsai integration is not complete yet; model and hardware requirements will be updated after the supported Bonsai runtime is selected and verified.

### Installation
```bash
poetry install
```

### Telegram Setup
To receive reports on your Telegram:
1.  **Create a Bot**: Message [@BotFather](https://t.me/BotFather) on Telegram and follow the steps to create a new bot and get your **API Token**.
2.  **Get Your Chat ID**:
    - Start a conversation with your new bot.
    - Message [@userinfobot](https://t.me/userinfobot) to get your unique **ID**.
3.  **Configure Environment**:
    - Copy `.env.example` to `.env`.
    - Fill in `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

### Usage

The platform can be run modularly or as a full automated pipeline.

#### 1. Full Automated Pipeline

Run everything from collection to delivery in one shot (by default, if no steps are specified, all steps run):

```bash
poetry run python scripts/run_pipeline.py --config config/pipeline_config.yaml --limit 10 --tiny
```

#### 2. Modular Steps
- **Collect News**: `python scripts/collect_news.py`
- **Preprocess Data**: `python scripts/preprocess_data.py`
- **Analyze Intelligence**: `python scripts/analyze_news.py --limit 5`
- **Generate Report**: `python scripts/generate_report.py`
- **Send to Telegram**: `python scripts/send_report.py`

## License
MIT
