"""
Prompt templates for news analysis tasks.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert financial and technology news analyst. "
    "Analyze the following article and provide a high-quality intelligence report. "
    "Respond ONLY in the requested format."
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """
Article:
{title}
{body}

Instruction: Provide a 2-3 sentence technical summary of the key developments in this article. Focus on 'the what' and 'the why'.
Summary:
"""

SENTIMENT_PROMPT = """
Article:
{title}
{body}

Instruction: Classify the market sentiment of this news.
Options: [Bullish, Bearish, Neutral]
Reasoning: Briefly explain why.
Output format: Sentiment: <Option> | Reasoning: <Explanation>
"""

IMPACT_PROMPT = """
Article:
{title}
{body}

Instruction: Assess the potential impact of this news on the relevant industry/market.
Scale: 1 (Minimal) to 10 (Critical)
Output format: Impact Score: <1-10> | Description: <One-sentence justification>
"""

ENTITIES_PROMPT = """
Article:
{title}
{body}

Instruction: Extract the top 5 key entities (Companies, People, Technologies, or Organizations) mentioned in this article.
Format: List them separated by commas.
Entities:
"""

# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------


def build_prompt(task: str, title: str, body: str) -> str:
    """Build a structured prompt for a specific task."""
    templates = {
        "summarize": SUMMARIZE_PROMPT,
        "sentiment": SENTIMENT_PROMPT,
        "impact": IMPACT_PROMPT,
        "entities": ENTITIES_PROMPT,
    }

    template = templates.get(task, SUMMARIZE_PROMPT)
    # Truncate body if too long for the prompt (basic safety)
    truncated_body = body[:2000] if len(body) > 2000 else body

    return (
        f"<s>[INST] <<SYS>>\n{_SYSTEM_PROMPT}\n<</SYS>>\n\n{template.format(title=title, body=truncated_body)} [/INST]"
    )
