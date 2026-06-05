"""E2E framework for vLLM PoC + cross-model validation testing.

Pipeline:
    deploy   — pull image, start container, wait /health
    poc      — collect PoC nonces via callback API
    infer    — run prompts_diverse.json against deployed model, save artifacts
    validate — replay model A's inference through model B with enforced_tokens
"""
