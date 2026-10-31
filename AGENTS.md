# AI News Intelligence Platform — Agent Instructions

These instructions apply to the whole repository. Follow them whenever working on this project.

## Project purpose and final scope

Build a dependable, portfolio-quality AI news intelligence platform that turns selected news sources into a traceable daily briefing. The mature system should:

1. Collect articles from configured RSS feeds and news APIs, recording source, publication time, fetch time, and canonical URL.
2. Clean, validate, normalize, and deduplicate articles while retaining the original source data and provenance.
3. Detect related coverage across outlets and group articles into stories or developments, rather than treating every article as an independent event.
4. Use a locally running Bonsai model for language-model tasks such as concise summarization, structured entity/topic extraction, classification, and impact assessment. Keep analysis grounded in source articles and links; label uncertainty and conflicting reports instead of presenting guesses as facts.
5. Prioritize developments for configured interests, regions, novelty, and likely impact. Keep sentiment as a limited signal, not a substitute for evidence or importance.
6. Produce a readable daily and on-demand briefing with story-level summaries, why each item matters, links to multiple sources, and changes since the previous briefing.
7. Deliver the briefing to the configured Telegram group, with clear success/failure reporting and safeguards against duplicate scheduled delivery.
8. Run the scheduled production workflow through GitHub Actions and the dedicated Kali self-hosted runner, with logs, useful failure visibility, and an auditable report artifact.
9. Be straightforward for a reviewer to set up and demonstrate from the README, with an understandable architecture, reproducible configuration, sample outputs, and documented limitations.

News source collection and Telegram delivery require network access. Model inference and text generation must use the local Bonsai runtime; do not silently substitute a hosted LLM or send article content to a cloud model. Keep any model-specific runtime behind a small interface so the rest of the pipeline does not depend on one inference engine.

Working model assumption: “Bonsai” means PrismML’s Bonsai family. Confirm the exact model variant, revision, license, supported inference runtime, and available machine hardware before selecting an integration. Do not assume every Bonsai variant works through standard Transformers or the same runtime. Pin the chosen model/runtime versions and measure memory, latency, output quality, and failure behavior on the actual target machine. See the [official Bonsai releases](https://prismml.com/news/bonsai-2-27b) for the model-family context.

## README phases and architecture alignment

The README's implementation phases and the project roadmap below describe different levels of work. Preserve the existing phases and modules; do not redo them as new standalone work. Extend and verify them through Steps 2–7. The code is present in this checkout, but presence alone does not mean it is production-ready or merged.

| README phase | Existing implementation in the repository | Remaining work tracked below |
|---|---|---|
| Phase 1 — Ingestion | `src/ingestion/` and `scripts/collect_news.py` collect from RSS and NewsAPI and persist raw articles. | Step 2 hardens failure handling, provenance, canonicalization, and deduplication. |
| Phase 2 — Data preprocessing | `src/data/` and `scripts/preprocess_data.py` clean article text and build a Hugging Face dataset. | Step 2 improves validation, repeatability, fixture-based checks, and provenance preservation. |
| Phase 3 — Intelligence/inference | `src/inference/`, `src/models/`, and `scripts/analyze_news.py` provide the inference structure. The README currently names Mistral-7B; the user's final requirement is local Bonsai inference. | Step 2 selects and validates a Bonsai variant/runtime for the target hardware, then integrates it behind the inference interface and measures quality, memory, and latency. |
| Phase 4 — Model experiments | `src/training/` and `scripts/train_tiny.py` provide experimental training scaffolding. | Keep training optional and separate from daily production unless evaluation demonstrates a need and a suitable labeled dataset exists. |
| Phase 5 — Aggregation and reporting | `src/reporting/` and report scripts provide basic aggregation and Markdown generation. | Steps 2 and 7 extend this to story grouping, evidence traceability, useful prioritization, and documented sample output. |
| Phase 6 — Telegram delivery | `src/delivery/` and `scripts/send_report.py` provide basic Telegram delivery. | Steps 3 and 6 verify real group delivery, clear failure status, retry behavior, and no duplicate scheduled message. |
| Phase 7 — Orchestration | `scripts/run_pipeline.py` coordinates the local pipeline. | Steps 4–6 add the daily GitHub Actions schedule, Kali runner, safe secrets, and observed end-to-end operation. |

The README architecture maps to the same modules: ingestion → `src/ingestion/`; processing → `src/data/`; inference → `src/inference/` plus `src/models/`; reporting → `src/reporting/`; delivery → `src/delivery/`; shared configuration/logging → `src/utils/`. Keep these boundaries understandable as they evolve. `src/training/` is present, but custom model training is not part of the committed final scope unless evaluation later shows a clear need and a suitable labeled dataset exists.
## Roadmap and acceptance checks

Step 1, repository workflow and CI foundation, is complete and merged into `main` as reported by the user. Keep `main` as the source of truth and continue using pull requests for focused changes.

### Step 2 — Make the local pipeline dependable

Improve predictable configuration, error handling, logging, repeatability, and data lineage across collection, preprocessing, analysis, reporting, and delivery. Separate external-service code from the processing logic. Add small fixtures or mocks where useful so routine checks do not require news credentials, Telegram, a GPU, or model downloads.

**Verify:** run the repository’s pinned lint and syntax checks; run the relevant automated checks for the changed stage; exercise a small fixture-based pipeline without external credentials or model downloads; confirm the report links back to its source articles.

### Step 3 — Make Telegram delivery verifiable

Confirm bot/group configuration, message formatting and splitting, file delivery, error handling, and clear CLI exit status. Store credentials outside Git. Preserve a manual way to validate connection and delivery.

**Verify:** with the user’s configured bot and intended group, run the CLI and confirm the report arrives in that group; verify invalid or missing configuration produces a clear failure. Do not claim a real Telegram send succeeded without observing the result.

### Step 4 — Add the daily GitHub Actions workflow

Create a scheduled workflow that collects, analyzes with the local Bonsai runtime, generates the briefing, and delivers it. Keep a manual trigger for controlled verification and recovery. Put secrets in GitHub Actions secrets, never in workflow files, logs, commits, or artifacts. Grant the workflow only the permissions it needs.

The target schedule is 08:00 Europe/Bucharest local time. GitHub cron uses UTC and does not follow daylight-saving changes. Use a DST-aware scheduling/guard approach so exactly one delivery occurs at 08:00 local time in both summer and winter; document and verify it.

**Verify:** manually trigger the workflow and confirm each stage in its logs, inspect the generated artifact, and confirm one Telegram delivery. Verify the schedule’s summer/winter behavior and that retries do not create duplicate messages.

### Step 5 — Set up the Kali PC as the self-hosted runner

Document installation, registration, labels, startup, updates, dependency/runtime setup, disk/memory checks, and recovery. Use a dedicated runner identity and repository-scoped access where possible. Do not run untrusted pull-request code on the self-hosted machine or expose production secrets to pull-request workflows. Keep the runner online only as needed and make runner status observable.

**Verify:** run a harmless runner-check workflow on Kali, confirm the expected labels and Python/Bonsai runtime are available, reboot the PC and confirm the runner returns online, and demonstrate that pull-request jobs cannot access the production runner or secrets.

### Step 6 — Verify the full scheduled path

Integrate the workflow, runner, local model, source collection, reporting, Telegram secrets, and failure visibility. Make job stages and outputs identifiable. Preserve a useful report artifact and enough logs to diagnose failures without exposing credentials or private message content.

**Verify:** run a controlled end-to-end Actions execution on Kali; check its run logs, report artifact, and actual Telegram delivery; then verify a scheduled run and a deliberately safe failure path notify the maintainer without sending a misleading or duplicate briefing.

### Step 7 — Polish the recruiter-facing repository

Document architecture and data flow, local setup, model/runtime choice, hardware requirements, configuration, CI, schedule, Kali runner setup, secrets, troubleshooting, sample briefing, evaluation method, and known limitations. Include sanitized evidence only; never commit real tokens, group identifiers that should remain private, private source data, model weights, or personal machine details.

**Verify:** follow the README from a clean checkout to a local demo; confirm documented commands work, required CI is green, the PR history tells a clear story, and a reviewer can distinguish implemented features from planned ones.

## Git and change-management rules

- Never put `codex` in branch names or commit messages.
- Keep `main` releasable. Use one short-lived branch per focused feature, fix, documentation, or CI change; open a pull request into `main` and wait for required checks before merging.
- Do not commit or push unrelated user work. Inspect status and diffs before staging; stage only files relevant to the requested change.
- Do not push until the relevant checks have passed locally. After pushing, inspect the GitHub check result and report whether it passed.
- Use clear conventional commit messages, such as `fix: handle malformed article files` or `ci: schedule daily briefing`.
- Never commit `.env`, API keys, Telegram tokens, private data, model weights, generated datasets, or machine-specific secrets.

## Learning-first collaboration

The user is learning AI engineering. Help them understand the work as well as receive the change.

- Explain the goal and the reason for a proposed change in plain language before or alongside implementation. Define AI/ML and infrastructure acronyms the first time they matter.
- Work in small, reviewable increments. For each step, describe what changed, why it matters, and how to verify it.
- Explain important choices and trade-offs, especially around model capability, hardware, local inference, evaluation, GitHub Actions, and runner security. Separate verified facts from assumptions.
- Prefer teaching the concept with the concrete code or command being changed. Avoid unexplained large rewrites and avoid dumping generic AI theory unrelated to the task.
- Maintain the roadmap and mark a step complete only when its stated verification has succeeded. If a check fails, report the exact cause and fix it before pushing.
- Do not claim tests, delivery, scheduling, or runner setup succeeded unless they were actually run and observed. Keep routine checks local and offline where possible; use real external services only for the verification the user requested.
- Make sensible implementation decisions autonomously. Ask for input only when a missing choice materially changes the design or an action would expose data, spend money, change security access, or make an irreversible external change.
