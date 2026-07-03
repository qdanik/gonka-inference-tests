"""`e2e poc-inference` — measure interference between PoC validation and inference.

Runs three phases against one vLLM server and saves a JSON per phase plus a
3-way comparison (table + timeline/bar plots):

    1. poc_only        — N validation requests, no inference (PoC baseline)
    2. inference_only  — sustained inference pool, no PoC (inference baseline)
    3. combined        — sustained inference pool + N validations concurrently

The headline signal is the inference *abort rate* under concurrent validation:
each PoC GPU batch aborts in-flight inference at the engine level
(vllm/poc/engine_patch.py), so the combined phase quantifies how badly the two
workloads disrupt each other.

Entry point: `python -m e2e.poc_inference run --ssh-host ... --model-name ...`.
"""
