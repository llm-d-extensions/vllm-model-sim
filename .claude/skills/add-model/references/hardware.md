# Hardware specs for the physics latency model

The source of truth is `catalog/hardware/<slug>.yaml`, one file per GPU.
`peak_tflops` is the **dense** BF16/FP16 tensor-core peak (vendor datasheets
often quote the 2× sparsity figure; don't use that). `hbm_gbps` is the memory
bandwidth in GB/s. These are per GPU; tensor parallelism is applied by the
plugin from `--tensor-parallel-size`.

| Slug | GPU | `peak_tflops` | `hbm_gbps` | vLLM defaults (batched tokens / seqs) |
|------|-----|---------------|------------|---------------------------------------|
| `h100-sxm5` | NVIDIA H100 SXM5 80 GB | 989 | 3350 | 8192 / 1024 |
| `h100-pcie` | NVIDIA H100 PCIe 80 GB | 756 | 2000 | 8192 / 1024 |
| `h200` | NVIDIA H200 141 GB | 989 | 4800 | 8192 / 1024 |
| `a100-sxm4-80gb` | NVIDIA A100 SXM4 80 GB | 312 | 2039 | 2048 / 256 |
| `a100-pcie-80gb` | NVIDIA A100 PCIe 80 GB | 312 | 1935 | 2048 / 256 |
| `l40s` | NVIDIA L40S 48 GB | 362 | 864 | 2048 / 256 |
| `b200` | NVIDIA B200 180 GB | 2250 | 8000 | 16384 / 1024 |

The vLLM defaults come from `EngineArgs.get_batch_defaults` in
`vllm/engine/arg_utils.py` (OpenAI server values): ≥ 160 GiB gets 16384/1024,
≥ 70 GiB non-A100 gets 8192/1024, and everything else gets 2048/256. The
renderer pins them on both the real and the sim pods, because on CPU vLLM would
otherwise default to 2048/128 per TP rank.

The physics model uses one `peak_tflops` for all math, whatever the weight
dtype. For FP8 checkpoints only the weight bytes change
(`weight_dtype: float8`) and the compute peak stays at the BF16 figure, so
FP8 prefill may come out slower than on real hardware running FP8 GEMMs.

AMD GPUs (e.g. MI300X: 1307 TFLOPs, 5300 GB/s) are not in the catalog: the
renderer requests `nvidia.com/gpu` for the real pods.

For hardware not listed, pass `--peak-tflops`, `--hbm-gbps`, `--memory-gib`
and `--gpu-product` to the scaffold script, which writes the catalog file.
Then add a row here.
