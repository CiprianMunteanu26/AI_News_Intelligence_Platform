# AI News Intelligence Platform — Task Checklist

## Phase 0 — Foundation & Project Setup ✅
- [x] Cleanup for git push (deleted ~61MB of artifacts)
- [x] Expand [.gitignore](file:///d:/projects/AI_News_Intelligence_Platform/.gitignore)
- [x] Add [__init__.py](file:///d:/projects/AI_News_Intelligence_Platform/src/__init__.py) files to all packages (10 files)
- [x] Add new dependencies to [pyproject.toml](file:///d:/projects/AI_News_Intelligence_Platform/pyproject.toml) (8 packages)
- [x] Create [src/utils/config.py](file:///d:/projects/AI_News_Intelligence_Platform/src/utils/config.py) + [config/pipeline_config.yaml](file:///d:/projects/AI_News_Intelligence_Platform/config/pipeline_config.yaml)
- [x] Create [src/utils/logger.py](file:///d:/projects/AI_News_Intelligence_Platform/src/utils/logger.py)
- [x] Create [.env.example](file:///d:/projects/AI_News_Intelligence_Platform/.env.example)

## Phase 1 — Data Ingestion Layer ✅
- [x] Create [src/ingestion/base_source.py](file:///d:/projects/AI_News_Intelligence_Platform/src/ingestion/base_source.py) — Article dataclass + NewsSource ABC
- [x] Create [src/ingestion/newsapi_source.py](file:///d:/projects/AI_News_Intelligence_Platform/src/ingestion/newsapi_source.py) — NewsAPI collector
- [x] Create [src/ingestion/rss_source.py](file:///d:/projects/AI_News_Intelligence_Platform/src/ingestion/rss_source.py) — RSS feed collector
- [x] Create [src/ingestion/collector.py](file:///d:/projects/AI_News_Intelligence_Platform/src/ingestion/collector.py) — orchestrator with dedup + daily JSON save
- [x] Create [scripts/collect_news.py](file:///d:/projects/AI_News_Intelligence_Platform/scripts/collect_news.py) — CLI entry point
- [x] Add BBC feeds (Top Stories, World, Technology, Business, Science)

## Phase 2 — Data Processing & Dataset Generation ✅
- [x] Implement [src/data/preprocess.py](file:///d:/projects/AI_News_Intelligence_Platform/src/data/preprocess.py)
- [x] Implement [src/data/dataset.py](file:///d:/projects/AI_News_Intelligence_Platform/src/data/dataset.py)
- [x] Create [scripts/preprocess_data.py](file:///d:/projects/AI_News_Intelligence_Platform/scripts/preprocess_data.py)

## Phase 3 — Intelligence Analysis & Inference ✅
- [x] Implement [src/models/factory.py](file:///d:/projects/AI_News_Intelligence_Platform/src/models/factory.py) (Model loading logic)
- [x] Create `src/inference/prompts.py` (Task-specific templates)
- [x] Implement [src/inference/engine.py](file:///d:/projects/AI_News_Intelligence_Platform/src/inference/engine.py) (Generation logic)
- [x] Implement [src/inference/analyzer.py](file:///d:/projects/AI_News_Intelligence_Platform/src/inference/analyzer.py) (Orchestration)
- [x] Create `scripts/analyze_news.py` (CLI for analysis)

## Phase 4 — Model Training Experiments (Optional)
- [x] Add tiny-model and training scaffolding under `src/training/` and `scripts/train_tiny.py`
- [ ] Keep training outside the production pipeline unless evaluation shows a need and a suitable labeled dataset is available

## Phase 5 — Intelligence Aggregation & Reporting ✅
- [x] Create `src/reporting/aggregator.py`
- [x] Create `src/reporting/report_generator.py`
- [x] Create `scripts/generate_report.py`

## Phase 6 — Telegram Delivery ✅
- [x] Create `src/delivery/telegram_bot.py`
- [x] Create `scripts/send_report.py`

## Phase 7 — Orchestration & Automation ✅
- [x] Create `scripts/run_pipeline.py`

## Phase 8 — Documentation ✅
- [x] Write comprehensive [README.md](file:///d:/projects/AI_News_Intelligence_Platform/README.md)

## Phase 9 — Verification ✅
- [x] Ruff static analysis across source, scripts, and tests
- [x] Python syntax compilation across source, scripts, and tests
- [x] Offline reporting and mocked Telegram delivery checks
- [ ] End-to-end run with the selected local Bonsai runtime, live news sources, and the configured Telegram group
