"""e2e/gateway — exercise the Gonka devshard gateway's request-validation layer.

Unlike the top-level `e2e infer` (which drives a raw vLLM server), this package
talks to the *gateway* (`/v1/chat/completions` on the server's loopback) to
verify how it handles chat-completion parameters — clamping, rejecting, or
normalizing out-of-range and wrong-typed values before they reach vLLM.

No secrets live here: the SSH host and admin key are supplied at runtime via
CLI flags or environment variables, never committed.
"""
