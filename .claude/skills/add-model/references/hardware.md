# Hardware specs for the physics latency model

`peak_tflops` is the **dense** BF16/FP16 tensor-core peak (vendor datasheets
often quote the 2× sparsity figure — don't use that). `hbm_gbps` is the
memory bandwidth in GB/s. These are per GPU; tensor parallelism is applied by
the plugin from `--tensor-parallel-size`.

| Slug | GPU | `peak_tflops` | `hbm_gbps` |
|------|-----|---------------|------------|
| `h100-sxm5` | NVIDIA H100 SXM5 80 GB | 989 | 3350 |
| `h100-pcie` | NVIDIA H100 PCIe 80 GB | 756 | 2000 |
| `h200` | NVIDIA H200 141 GB | 989 | 4800 |
| `a100-sxm4-80gb` | NVIDIA A100 SXM4 80 GB | 312 | 2039 |
| `a100-pcie-80gb` | NVIDIA A100 PCIe 80 GB | 312 | 1935 |
| `l40s` | NVIDIA L40S 48 GB | 362 | 864 |
| `b200` | NVIDIA B200 | 2250 | 8000 |
| `mi300x` | AMD Instinct MI300X | 1307 | 5300 |

The physics model uses one `peak_tflops` for all math, whatever the weight
dtype. For FP8 checkpoints only the weight bytes change
(`weight_dtype: float8`) and the compute peak stays at the BF16 figure, so
FP8 prefill may come out slower than on real hardware running FP8 GEMMs.

For hardware not listed, pass `--peak-tflops` and `--hbm-gbps` to the
scaffold script, and add a row here and to the "Hardware Slugs" table in
`models/README.md`.
