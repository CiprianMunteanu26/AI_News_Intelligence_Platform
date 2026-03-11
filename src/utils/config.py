"""
Centralized configuration for the AI News Intelligence Platform.

Loads settings from config/pipeline_config.yaml and .env variables.
Provides typed access to all pipeline parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root (two levels up from src/utils/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    analysis_dir: Path = Path("data/analysis")
    checkpoints_dir: Path = Path("checkpoints")
    reports_dir: Path = Path("reports")

    def resolve(self, root: Path) -> None:
        """Convert relative paths to absolute paths based on project root."""
        for attr in ("raw_data_dir", "processed_data_dir", "analysis_dir", "checkpoints_dir", "reports_dir"):
            path = Path(getattr(self, attr))
            if not path.is_absolute():
                path = root / path
            setattr(self, attr, path)

    def ensure_dirs(self) -> None:
        """Create all configured directories if they don't exist."""
        for attr in ("raw_data_dir", "processed_data_dir", "analysis_dir", "checkpoints_dir", "reports_dir"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionSourceConfig:
    type: str = "newsapi"
    enabled: bool = True
    categories: list[str] = field(default_factory=lambda: ["technology", "business"])
    country: str = "us"
    page_size: int = 50
    feeds: list[str] = field(default_factory=list)


@dataclass
class IngestionConfig:
    sources: list[IngestionSourceConfig] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    min_article_length: int = 100
    max_article_length: int = 10000
    dedup_similarity_threshold: float = 0.85
    language: str = "en"


@dataclass
class TrainingConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    fp16: bool = True


@dataclass
class InferenceConfig:
    batch_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.3
    tasks: list[str] = field(default_factory=lambda: ["summarize", "sentiment", "impact", "entities"])


@dataclass
class ReportingConfig:
    top_n_articles: int = 20
    sections: list[str] = field(
        default_factory=lambda: [
            "Top Developments",
            "Key Signals",
            "Technology Highlights",
            "Market Signals",
            "Geopolitical Events",
        ]
    )


@dataclass
class ScheduleConfig:
    collect_interval_hours: int = 1
    preprocess_interval_hours: int = 3
    inference_time: str = "06:00"
    report_time: str = "07:00"


# ---------------------------------------------------------------------------
# Main platform config
# ---------------------------------------------------------------------------
@dataclass
class PlatformConfig:
    """Top-level configuration object for the entire platform."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    # Environment secrets (loaded from .env, not from YAML)
    newsapi_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------
    @classmethod
    def load(cls, config_path: Path | str | None = None) -> PlatformConfig:
        """
        Load configuration from YAML + environment variables.

        Parameters
        ----------
        config_path : optional path to YAML config; defaults to config/pipeline_config.yaml
        """
        load_dotenv(PROJECT_ROOT / ".env")

        config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        cfg = cls(
            paths=_build(PathsConfig, raw.get("paths", {})),
            ingestion=_build_ingestion(raw.get("ingestion", {})),
            preprocessing=_build(PreprocessingConfig, raw.get("preprocessing", {})),
            training=_build(TrainingConfig, raw.get("training", {})),
            inference=_build(InferenceConfig, raw.get("inference", {})),
            reporting=_build(ReportingConfig, raw.get("reporting", {})),
            schedule=_build(ScheduleConfig, raw.get("schedule", {})),
            newsapi_key=os.getenv("NEWSAPI_KEY", ""),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        # Env override for model name
        env_model = os.getenv("MODEL_NAME")
        if env_model:
            cfg.training.model_name = env_model

        # Resolve & ensure paths
        cfg.paths.resolve(PROJECT_ROOT)
        cfg.paths.ensure_dirs()

        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build(cls, data: dict) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def _build_ingestion(data: dict) -> IngestionConfig:
    sources = []
    for src in data.get("sources", []):
        sources.append(_build(IngestionSourceConfig, src))
    return IngestionConfig(sources=sources)
