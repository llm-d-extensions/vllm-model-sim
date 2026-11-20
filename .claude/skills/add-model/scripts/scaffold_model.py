#!/usr/bin/env python3
"""Scaffold models/<slug>/ with an untuned physics deployment.

Writes:
  models/<slug>/config.json                      real HF config (cleaned)
  models/<slug>/deployments/<hw>/standalone/k8s/ sim-config.json, configmap.yaml,
                                                 deployment.yaml, service.yaml, pvc.yaml

The latency block is the physics roofline model with unit betas
[1.0, 1.0, 0.0] — no tuning. READMEs and inventory tables are left to the
caller (see SKILL.md); this script only produces the files whose contents
are mechanical.

Run from the repo root:
  python .claude/skills/add-model/scripts/scaffold_model.py \
      --model Qwen/Qwen3-8B --slug qwen3-8b --hardware h100-sxm5 [--tp 1]
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys

# Dense BF16 tensor-core peak (no sparsity) and HBM bandwidth per GPU.
HARDWARE = {
    "h100-sxm5": (989.0, 3350.0),
    "h100-pcie": (756.0, 2000.0),
    "h200": (989.0, 4800.0),
    "a100-sxm4-80gb": (312.0, 2039.0),
    "a100-pcie-80gb": (312.0, 1935.0),
    "l40s": (362.0, 864.0),
    "b200": (2250.0, 8000.0),
    "mi300x": (1307.0, 5300.0),
}

DEFAULT_BETA = [1.0, 1.0, 0.0]
# Keys the physics model and vLLM's config loader read from the top level.
ARCH_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "decoder_sparse_step",
    "eos_token_id",
    "torch_dtype",
)
# Keys that make vLLM do real work (quantized kernels, remote code) the sim
# cannot honour with dummy weights.
DROP_KEYS = ("quantization_config", "auto_map", "compression_config")
# Markers of architectures the physics model does not represent faithfully.
UNSUPPORTED = {
    "kv_lora_rank": "MLA attention (DeepSeek-style) — KV size and FLOPs will be wrong",
    "linear_attention": "hybrid linear attention — treated as full attention",
    "layer_types": "per-layer attention types — treated as uniform full attention",
    "sliding_window": "sliding-window attention — modelled as full attention",
    "vision_config": "multimodal model — only the text tower is simulated",
}
QUANT_DTYPE = {"fp8": "float8", "int8": "int8"}


def load_config(model: str) -> dict:
    if os.path.isfile(model):
        with open(model) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # Plain HTTPS works for public repos; gated ones need huggingface_hub
        # (or HF_TOKEN below).
        import urllib.request

        req = urllib.request.Request(
            f"https://huggingface.co/{model}/resolve/main/config.json"
        )
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    with open(hf_hub_download(repo_id=model, filename="config.json")) as f:
        return json.load(f)


def clean_real_config(raw: dict) -> tuple[dict, str, list[str]]:
    """Return (config, weight_dtype, warnings) for the committed config.json."""
    cfg = copy.deepcopy(raw)
    warnings = []

    # Multimodal / wrapper configs keep the decoder under text_config; the
    # physics model reads attributes from the top level, so hoist them.
    text = cfg.get("text_config") or cfg.get("language_config")
    if isinstance(text, dict) and "hidden_size" not in cfg:
        for k in ARCH_KEYS:
            if k in text and k not in cfg:
                cfg[k] = text[k]
        warnings.append("hoisted decoder fields from text_config to the top level")

    # Mixtral names it num_local_experts; the physics model reads num_experts.
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]

    for key, why in UNSUPPORTED.items():
        if cfg.get(key) or (isinstance(text, dict) and text.get(key)):
            if key == "layer_types" and set(cfg.get(key) or []) <= {"full_attention"}:
                continue
            warnings.append(f"{key}: {why}")

    if cfg.get("num_experts") and cfg.get("decoder_sparse_step") == 1:
        warnings.append(
            "decoder_sparse_step=1: HF means every layer is MoE, but the physics "
            "model counts only half the layers as MoE (L // (step + 1))"
        )

    weight_dtype = "bfloat16"
    q = cfg.get("quantization_config") or {}
    qm = str(q.get("quant_method", "")).lower()
    if qm:
        weight_dtype = QUANT_DTYPE.get(qm, weight_dtype)
        if qm not in QUANT_DTYPE:
            warnings.append(
                f"quant_method {qm!r} has no physics weight_dtype; using bfloat16"
            )
    elif cfg.get("torch_dtype") == "float16":
        weight_dtype = "float16"
    for k in DROP_KEYS:
        cfg.pop(k, None)
    cfg.pop("latency", None)

    missing = [k for k in ("hidden_size", "num_hidden_layers", "num_attention_heads") if k not in cfg]
    if missing:
        sys.exit(f"config is missing required decoder fields: {missing}")
    return cfg, weight_dtype, warnings


def build_sim_config(real: dict, peak_tflops: float, hbm_gbps: float, weight_dtype: str,
                     beta=DEFAULT_BETA) -> dict:
    sim = copy.deepcopy(real)
    sim["architectures"] = ["SimulatedForCausalLM"]
    # tp is injected at load time from --tensor-parallel-size; never write it.
    sim["latency"] = {
        "type": "physics",
        "hardware": {
            "peak_tflops": peak_tflops,
            "hbm_gbps": hbm_gbps,
            "weight_dtype": weight_dtype,
        },
        "beta": [float(b) for b in beta],
        "deterministic_length": True,
    }
    return sim


def config_hash(sim: dict) -> str:
    # Same scheme as the existing manifests: sha256 of the sorted-key JSON.
    return hashlib.sha256(json.dumps(sim, sort_keys=True).encode()).hexdigest()[:6]


def render_configmap(name: str, sim_json: str) -> str:
    body = "\n".join("    " + line for line in sim_json.splitlines())
    return (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n"
        f"  name: {name}-config\ndata:\n  config.json: |\n{body}\n"
    )


def render_deployment(name, slug, served, hf_id, tp, max_len) -> str:
    cpu = str(4 * tp)
    mem = f"{8 * tp}Gi"
    omp = f"0-{4 * tp - 1}"
    tp_arg = f"        - --tensor-parallel-size={tp}\n" if tp > 1 else ""
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      initContainers:
      - name: install-plugin
        image: ghcr.io/llm-d-extensions/vllm-sim-deps:v0.3.0
        imagePullPolicy: Always
        command:
        - sh
        - -c
        - cp -r /plugins/* /mnt/plugins/
        volumeMounts:
        - name: plugins
          mountPath: /mnt/plugins
      containers:
      - name: vllm
        image: vllm/vllm-openai-cpu:v0.27.0-x86_64
        command:
        - vllm
        - serve
        - /model
        - --load-format=dummy
        - --max-model-len={max_len}
        - --tokenizer={hf_id}
        - --served-model-name={served}
        - --gpu-memory-utilization=0.5
{tp_arg}        env:
        - name: PYTHONPATH
          value: /plugins
        - name: VLLM_CPU_KVCACHE_SPACE
          value: "4"
        - name: VLLM_CPU_OMP_THREADS_BIND
          value: "{omp}"
        - name: HOME
          value: /tmp
        - name: XDG_CACHE_HOME
          value: /tmp/cache
        - name: HF_HOME
          value: /hf-cache
        - name: HF_HUB_CACHE
          value: /hf-cache
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
              optional: true
        ports:
        - containerPort: 8000
          name: http
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 5
          failureThreshold: 30
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            cpu: "{cpu}"
            memory: {mem}
          limits:
            cpu: "{cpu}"
            memory: {mem}
        volumeMounts:
        - name: plugins
          mountPath: /plugins
        - name: model-config
          mountPath: /model
          readOnly: true
        - name: dshm
          mountPath: /dev/shm
        - name: hf-cache
          mountPath: /hf-cache
      volumes:
      - name: plugins
        emptyDir: {{}}
      - name: model-config
        configMap:
          name: {name}-config
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 256Mi
      - name: hf-cache
        persistentVolumeClaim:
          claimName: vllm-{slug}-hf-cache
"""


def render_service(name: str) -> str:
    return f"""apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP
"""


def render_pvc(slug: str) -> str:
    return f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-{slug}-hf-cache
spec:
  # ReadWriteMany so every replica shares one warm HuggingFace cache: the
  # first pod downloads the tokenizer, the rest read it without hitting the
  # network on the cold-start critical path.
  accessModes:
    - ReadWriteMany
  storageClassName: ibm-spectrum-scale-fileset
  resources:
    requests:
      storage: 1Gi
"""


def write(path: str, text: str, force: bool) -> None:
    if os.path.exists(path) and not force:
        sys.exit(f"refusing to overwrite {path} (pass --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF id, or path to a config.json")
    ap.add_argument("--hf-id", help="HF id for --tokenizer (required if --model is a path)")
    ap.add_argument("--slug", required=True,
                    help="directory and resource-name stem, DNS-safe, e.g. llama-3-1-8b-instruct")
    ap.add_argument("--served-model-name", help="API model name (default: slug); dots allowed")
    ap.add_argument("--hardware", default="h100-sxm5", help=f"one of {sorted(HARDWARE)} or custom")
    ap.add_argument("--peak-tflops", type=float, help="override (required for custom hardware)")
    ap.add_argument("--hbm-gbps", type=float, help="override (required for custom hardware)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--beta", type=float, nargs=3, default=DEFAULT_BETA,
                    metavar=("PF", "DC", "BASE_MS"), help="default: 1 1 0 (untuned)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--root", default="models")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", args.slug):
        sys.exit(f"slug {args.slug!r} must be lower-case letters, digits and '-' "
                 "(it ends up in Kubernetes resource names); e.g. replace '.' with '-'")
    served = args.served_model_name or args.slug
    hf_id = args.hf_id or (None if os.path.isfile(args.model) else args.model)
    if not hf_id:
        sys.exit("--hf-id is required when --model is a local path")
    peak, bw = HARDWARE.get(args.hardware, (None, None))
    peak = args.peak_tflops or peak
    bw = args.hbm_gbps or bw
    if peak is None or bw is None:
        sys.exit(f"unknown hardware {args.hardware!r}: pass --peak-tflops and --hbm-gbps")

    real, weight_dtype, warnings = clean_real_config(load_config(args.model))
    sim = build_sim_config(real, peak, bw, weight_dtype, args.beta)
    h = config_hash(sim)
    name = f"vllm-{args.slug}-standalone-{h}"
    max_len = min(args.max_model_len, real.get("max_position_embeddings", args.max_model_len))

    model_dir = os.path.join(args.root, args.slug)
    k8s = os.path.join(model_dir, "deployments", args.hardware, "standalone", "k8s")
    sim_json = json.dumps(sim, indent=2) + "\n"
    write(os.path.join(model_dir, "config.json"), json.dumps(real, indent=2) + "\n", args.force)
    write(os.path.join(k8s, "sim-config.json"), sim_json, args.force)
    write(os.path.join(k8s, "configmap.yaml"), render_configmap(name, sim_json), args.force)
    write(os.path.join(k8s, "deployment.yaml"),
          render_deployment(name, args.slug, served, hf_id, args.tp, max_len), args.force)
    write(os.path.join(k8s, "service.yaml"), render_service(name), args.force)
    write(os.path.join(k8s, "pvc.yaml"), render_pvc(args.slug), args.force)

    summary = {
        "resource_name": name,
        "hash": h,
        "hf_id": hf_id,
        "served_model_name": served,
        "hardware": {"slug": args.hardware, "peak_tflops": peak, "hbm_gbps": bw,
                     "weight_dtype": weight_dtype},
        "tp": args.tp,
        "max_model_len": max_len,
        "beta": sim["latency"]["beta"],
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
