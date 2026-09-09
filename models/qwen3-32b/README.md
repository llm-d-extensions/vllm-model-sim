# Qwen3-32B

Dense transformer, 32 billion parameters.

- **HuggingFace**: `Qwen/Qwen3-32B`
- **Architecture**: `Qwen3ForCausalLM`
- **Parameters**: 32 B
- **Max context**: 32 K tokens (40 960 max position embeddings)
- **dtype**: bfloat16

## Deployments

| Directory | Hardware | TP | Configurations | Latency models | Eval results |
|-----------|----------|----|----------------|----------------|--------------|
| [h100-sxm5](deployments/h100-sxm5/README.md) | H100 SXM5 80 GB | 1 (standalone, pd), 2 (pd-tp2) | standalone, pd, pd-tp2 | flat, physics, physics-beta-1.0 — see the hardware README for per-deployment beta values | flat, physics, physics-beta-1.0 (pd-tp2: flat, physics-beta-1.0 only — physics not yet run) |
