"""Training-data generation helpers for Lanty.

Split from the original generate_training_data.py so no single file exceeds
the project's 700-line cap. The top-level script composes these modules:

    topics   — player-prompt pools per dialogue mode
    prompts  — system/user prompt builders + context loaders
    api      — Anthropic client call + per-batch worker
"""
