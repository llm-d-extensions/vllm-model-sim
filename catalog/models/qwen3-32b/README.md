# Qwen3-32B

Dense transformer, 32 billion parameters.

- **HuggingFace**: `Qwen/Qwen3-32B`
- **Architecture**: `Qwen3ForCausalLM`
- **Parameters**: 32 B
- **Max context**: 32 K tokens (40 960 max position embeddings)
- **dtype**: bfloat16

## Stacks

| Stack | Hardware | Topology | TP |
|-------|----------|----------|----|
| [qwen3-32b-h100-sxm5-standalone](../../../deploy/qwen3-32b-h100-sxm5-standalone/) | H100 SXM5 80 GB | standalone | 1 |
| [qwen3-32b-h100-sxm5-pd](../../../deploy/qwen3-32b-h100-sxm5-pd/) | H100 SXM5 80 GB | P/D | 1 |
| [qwen3-32b-h100-sxm5-pd-tp2](../../../deploy/qwen3-32b-h100-sxm5-pd-tp2/) | H100 SXM5 80 GB | P/D | 2 |

Calibrations: [calibrations/qwen3-32b/](../../../calibrations/qwen3-32b/).
