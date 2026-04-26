"""Anthropic API call + per-batch worker.

The system prompt (~100KB of lore/personality/samples) is passed with
cache_control so the first call pays a ~1.25x write cost and every subsequent
call reads the cache at ~0.1x. That's what makes 100+ batches feasible on
Tier 1 rate limits.
"""

import time

import anthropic

from .prompts import build_multi_turn_prompt, build_single_turn_prompt

MODEL = "claude-sonnet-4-6"


def make_api_call(
    client: anthropic.Anthropic,
    system: str,
    prompt: str,
    max_tokens: int = 8000,
    retries: int = 6,
) -> tuple[str, dict]:
    """Make a single API call with retry logic. Returns (text, usage_dict).

    Rate limit errors (429) get longer backoff because Anthropic's rate limit
    windows are 60s — short retries are useless. Other errors get exponential
    backoff up to ~30s.
    """
    last_error = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )
            usage = {
                "input": response.usage.input_tokens,
                "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                "output": response.usage.output_tokens,
            }
            return response.content[0].text, usage
        except anthropic.RateLimitError as e:
            last_error = e
            wait = min(30 + 15 * attempt, 90)
            time.sleep(wait)
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
    raise last_error


def generate_batch(
    client: anthropic.Anthropic,
    system: str,
    mode: str,
    topics: list[str],
    count: int,
) -> tuple[str, str, dict]:
    """Generate a batch and return (mode, content, usage)."""
    if mode == "multi_turn":
        prompt = build_multi_turn_prompt(count)
    else:
        prompt = build_single_turn_prompt(topics, mode, count)

    result, usage = make_api_call(client, system, prompt)
    return mode, result, usage
