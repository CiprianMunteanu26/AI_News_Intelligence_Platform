# AI News Intelligence Platform

An automated pipeline for collecting and analyzing news with a locally run language model.

## Overview

The platform collects news from configured RSS feeds and NewsAPI, preprocesses articles, analyzes them with a local PrismML Bonsai model, and saves structured analysis results.

## Architecture

1. **Ingestion** — `src/ingestion/` collects and stores raw articles.
2. **Processing** — `src/data/` cleans articles and builds a Hugging Face dataset.
3. **Inference** — `src/inference/` sends task prompts to the local Bonsai runtime.
4. **Model experiments** — `src/training/` contains optional experimental training code.

## Requirements

- Python 3.11 or newer and Poetry
- A local `llama.cpp` server build with Bonsai 2 support
- A Bonsai GGUF model kept outside the repository
- Optional NewsAPI credentials

The configured baseline is **Bonsai 2 2B Q2_0**, selected to keep inference practical on the 4 GB GTX 1050 Ti planned for release execution. Its weights are approximately 0.55 GB; total runtime memory also depends on context size and backend. Bonsai 2 9B can be evaluated on the RX 6700 XT machine, but it is not the release target.

## Install

```powershell
poetry install
Copy-Item .env.example .env
```

Set `NEWSAPI_KEY` in `.env` if using NewsAPI. Never commit `.env`.

## Start the Bonsai runtime

Download the Apache-2.0 Bonsai 2 2B Q2_0 weights from the [official model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf). This repository checked model revision `683f7be2629342a50987627b40881e54761a2430`. With Hugging Face CLI installed, download that exact revision with:

```bash
poetry run hf download prism-ml/Ternary-Bonsai-2-2B-gguf Ternary-Bonsai-2-2B-Q2_0.gguf --revision 683f7be2629342a50987627b40881e54761a2430 --local-dir ./models
```

Use the PrismML `llama.cpp` release pinned by the [official backend support notes](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md) ([`prism-b10828-a153697`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10828-a153697) at the time this setup was checked), or a newer build that supports the Bonsai 2 Q2_0 format. Keep model weights and runtime binaries outside Git.

Start the server on the same machine as the pipeline:

```bash
llama-server -m ./models/Ternary-Bonsai-2-2B-Q2_0.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -fa on
```

On Windows, run `llama-server.exe` with the same arguments. `-ngl 99` requests GPU offload; use `-ngl 0` for CPU inference if the GPU backend is unavailable. GPU support, speed, and memory use depend on the exact card, driver, runtime build, and context size. Validate on the actual release machine before relying on scheduled runs.

The default API base URL is `http://127.0.0.1:8080/v1`. Override it with `BONSAI_RUNTIME_URL` in `.env` or `inference.runtime_url` in `config/pipeline_config.yaml`. Only loopback URLs are accepted, so article text is not sent to a remote model service.

Check that the server has loaded the model:

```powershell
poetry run python scripts/analyze_news.py --config config/pipeline_config.yaml --check-runtime
```

## Run the analysis stages

```powershell
poetry run python scripts/collect_news.py --config config/pipeline_config.yaml
poetry run python scripts/preprocess_data.py --config config/pipeline_config.yaml
poetry run python scripts/analyze_news.py --config config/pipeline_config.yaml --limit 3
```

The preprocessing command builds `latest_news_dataset` from only the most recently modified JSON collection in `data/raw/`; it does not combine old collections. Run collection before preprocessing so this points at the current run. The analysis command uses Bonsai by default and writes results under `data/analysis/`. Start the local model server before running it. `--tiny` selects the small Transformers fallback instead of Bonsai; it may download model files and is not the production inference path.

Collection can fill short RSS or NewsAPI excerpts from the publisher page when its HTTPS hostname matches `ingestion.article_domains`. Fetches have a 10-second timeout, are limited to 10 pages per source, follow at most three HTTPS redirects that remain on configured domains, and read at most 2 MiB per page. The original feed excerpt is retained in article metadata. Adjust or disable this behavior in `config/pipeline_config.yaml`; pages outside the allowlist remain snippets.

Analysis abstains without making model calls when an article body is shorter than `inference.minimum_body_characters` (500 by default). Task prompts request concise, evidence-based formats; analysis results remove explicit `<think>` blocks and reject malformed sentiment or impact responses instead of storing them.

## Offline checks

```powershell
poetry run ruff check --extend-select W291,W293 src scripts tests
poetry run python -m compileall -q src scripts tests
poetry run python -m unittest discover -s tests -v
```

The inference tests mock the local HTTP API and do not require model weights, a running server, or news credentials.

## License

MIT
