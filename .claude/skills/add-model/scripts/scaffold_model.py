#!/usr/bin/env python3
"""Scaffold a catalog model and a standalone stack with untuned physics.

Writes:
  catalog/models/<slug>/config.json    real HF config (cleaned)
  catalog/models/<slug>/model.yaml     HF id, served model name, weight dtype
  catalog/hardware/<hw>.yaml           only for new hardware (--peak-tflops ...)
  calibrations/<slug>/<hw>.yaml        flat (linear) debugging record; --beta record
  deploy/<stack>/stack.yaml, sweep.yaml
  deploy/<stack>/k8s/                  rendered by `python -m simstack render`

With no physics calibration record for the stack, the sim runs the physics
roofline model with unit betas [1.0, 1.0, 0.0], no tuning. If
catalog/models/<slug>/ exists, it is reused and --model may be omitted. READMEs and inventory tables
are left to the caller (see SKILL.md); this script only produces the files
whose contents are mechanical.

Run from the repo root (needs PyYAML, as simstack does):
  python .claude/skills/add-model/scripts/scaffold_model.py \
      --model Qwen/Qwen3-8B --slug qwen3-8b --hardware h100-sxm5 [--tp 1]
"""

import argparse
import copy
import json
import os
import re
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)

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

    # Mixtral and gpt-oss name it num_local_experts and give the per-expert
    # FFN width as intermediate_size; the physics model reads num_experts and
    # moe_intermediate_size, and would otherwise cost the MoE FFN at zero.
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]
    if cfg.get("num_experts") and "moe_intermediate_size" not in cfg:
        cfg["moe_intermediate_size"] = cfg.get("intermediate_size", 0)
        warnings.append("moe_intermediate_size: copied from intermediate_size "
                        "(per-expert width, Mixtral-style config)")

    for key, why in UNSUPPORTED.items():
        if cfg.get(key) or (isinstance(text, dict) and text.get(key)):
            layers = cfg.get(key) or (text or {}).get(key) or []
            if key == "layer_types" and set(layers) <= {"full_attention"}:
                continue
            if key == "sliding_window" and cfg.get("use_sliding_window") is False:
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


def vllm_defaults(memory_gib: float, name: str) -> dict:
    """vLLM's OpenAI-server scheduler defaults for a GPU.

    Mirrors EngineArgs.get_batch_defaults in vllm/engine/arg_utils.py.
    """
    if memory_gib >= 160:
        return {"max_num_batched_tokens": 16384, "max_num_seqs": 1024}
    if memory_gib >= 70 and "a100" not in name.lower():
        return {"max_num_batched_tokens": 8192, "max_num_seqs": 1024}
    return {"max_num_batched_tokens": 2048, "max_num_seqs": 256}


def write(path: str, text: str, force: bool) -> None:
    if os.path.exists(path) and not force:
        sys.exit(f"refusing to overwrite {path} (pass --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(path, REPO)}")


def write_hardware(path, args) -> None:
    d = vllm_defaults(args.memory_gib, args.hardware)
    write(path, f"""# {args.hardware} (added by the add-model skill; check the specs).
name: {args.hardware}
peak_tflops: {args.peak_tflops}        # dense BF16
hbm_gbps: {args.hbm_gbps}
memory_gib: {args.memory_gib:g}
# Node selector for the real GPU pods. Verify or replace for your cluster.
node_selector:
  nvidia.com/gpu.product: {args.gpu_product}
# vLLM's scheduler defaults for this GPU class (OpenAI server; see
# EngineArgs.get_batch_defaults in vllm/engine/arg_utils.py).
vllm_defaults:
  max_num_batched_tokens: {d["max_num_batched_tokens"]}
  max_num_seqs: {d["max_num_seqs"]}
# KV transfer between prefill and decode pods (simulated NIXL).
kv_transfer:
  bandwidth_gbps: 100
  handshake_ms: 2
""", args.force)


CALIBRATION_HEADER = """\
# Latency calibrations for {model} on {hw}.
#
# A record applies to a sim pod when every key in its `match` equals the pod's
# value (keys: topology, tp, role). The most specific match wins; a tie is an
# error. With no physics match, the sim falls back to unit betas [1.0, 1.0, 0.0].
#
# method: fitted (tuned against the real GPU), copied (taken from another
# record, never measured for this match) or assumed.
"""
# The `flat` eval variant needs a linear record; without one eval.sh and
# tests/test_simstack.py fail for the stack. Same values as Qwen3-32B's.
FLAT_RECORD = {
    "latency": {"type": "linear", "base_ms": 5.0, "prefill_ms_per_token": 0.05,
                "decode_ms_per_seq": 1.2, "ctx_ms_per_ktoken": 0.3},
    "method": "assumed",
    "note": "Flat latency model, a debugging aid; not fitted.",
}


def write_calibrations(args) -> None:
    """Ensure a linear record exists; upsert a physics record for --beta.

    Physics is left unrecorded by default so the stack resolves to unit betas.
    """
    import yaml

    from simstack.__main__ import _header, dump_calibrations
    from simstack.catalog import calibration_path

    path = calibration_path(args.slug, args.hardware)
    if path.exists():
        text = path.read_text()
        header, records = _header(text), yaml.safe_load(text) or []
    else:
        header = CALIBRATION_HEADER.format(model=args.slug, hw=args.hardware)
        records = []
    changed = False
    if not any(r["latency"].get("type") == "linear" for r in records):
        records.append(copy.deepcopy(FLAT_RECORD))
        changed = True
    if args.beta:
        match = {"topology": "standalone", "tp": args.tp}
        record = {"match": match,
                  "latency": {"type": "physics", "beta": [float(b) for b in args.beta]},
                  "method": "assumed", "note": "Supplied by the user, not fitted."}
        for i, r in enumerate(records):
            if r.get("match") == match and r["latency"].get("type") == "physics":
                records[i] = record
                break
        else:
            records.insert(0, record)
        changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n" + dump_calibrations(records))
        print(f"wrote {os.path.relpath(path, REPO)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model",
                    help="HF id, or path to a config.json (omit to reuse catalog/models/<slug>/)")
    ap.add_argument("--hf-id", help="HF id for --tokenizer (required if --model is a path)")
    ap.add_argument("--slug", required=True,
                    help="catalog name and resource-name stem, DNS-safe, e.g. llama-3-1-8b-instruct")
    ap.add_argument("--served-model-name", help="API model name (default: slug); dots allowed")
    ap.add_argument("--hardware", default="h100-sxm5",
                    help="a catalog/hardware/ slug, or a new one with --peak-tflops etc.")
    ap.add_argument("--peak-tflops", type=float, help="dense BF16 peak, for new hardware")
    ap.add_argument("--hbm-gbps", type=float, help="HBM bandwidth (GB/s), for new hardware")
    ap.add_argument("--memory-gib", type=float, help="GPU memory (GiB), for new hardware")
    ap.add_argument("--gpu-product", help="nvidia.com/gpu.product label, for new hardware")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--beta", type=float, nargs=3, metavar=("PF", "DC", "BASE_MS"),
                    help="user-supplied betas (default: none, i.e. unit betas)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", args.slug):
        sys.exit(f"slug {args.slug!r} must be lower-case letters, digits and '-' "
                 "(it ends up in Kubernetes resource names); e.g. replace '.' with '-'")

    hw_path = os.path.join(REPO, "catalog", "hardware", f"{args.hardware}.yaml")
    new_hw = not os.path.exists(hw_path)
    if new_hw and None in (args.peak_tflops, args.hbm_gbps, args.memory_gib,
                           args.gpu_product):
        sys.exit(f"unknown hardware {args.hardware!r}: pass --peak-tflops, "
                 "--hbm-gbps, --memory-gib and --gpu-product to add it")

    # An existing catalog model is reused as is (new GPU or TP for it), unless
    # --model and --force ask to replace it.
    model_dir = os.path.join(REPO, "catalog", "models", args.slug)
    reuse = os.path.exists(os.path.join(model_dir, "model.yaml")) and not (
        args.model and args.force)
    if reuse:
        from simstack.catalog import load_model

        meta, _ = load_model(args.slug)
        hf_id, served = meta["hf_id"], meta["served_model_name"]
        weight_dtype, warnings = meta.get("weight_dtype", "bfloat16"), []
        print(f"reusing catalog/models/{args.slug}/")
    else:
        if not args.model:
            sys.exit(f"catalog/models/{args.slug}/ does not exist: pass --model")
        served = args.served_model_name or args.slug
        hf_id = args.hf_id or (None if os.path.isfile(args.model) else args.model)
        if not hf_id:
            sys.exit("--hf-id is required when --model is a local path")
        # Load before writing anything, so a failed download leaves no files.
        real, weight_dtype, warnings = clean_real_config(load_config(args.model))

    if new_hw:
        write_hardware(hw_path, args)
    if not reuse:
        write(os.path.join(model_dir, "config.json"),
              json.dumps(real, indent=2) + "\n", args.force)
        write(os.path.join(model_dir, "model.yaml"),
              f"hf_id: {hf_id}\nserved_model_name: {served}\n"
              f"weight_dtype: {weight_dtype}\n", args.force)

    stack = f"{args.slug}-{args.hardware}-standalone" + (
        f"-tp{args.tp}" if args.tp > 1 else "")
    stack_dir = os.path.join(REPO, "deploy", stack)
    write(os.path.join(stack_dir, "stack.yaml"),
          f"model: {args.slug}\nhardware: {args.hardware}\ntopology: standalone\n"
          f"tp: {args.tp}\nprofile: default\n", args.force)
    sweep = os.path.join(stack_dir, "sweep.yaml")
    if not os.path.exists(sweep) or args.force:
        shutil.copy(os.path.join(REPO, "evaluation", "sweep.yaml"), sweep)

    write_calibrations(args)

    from simstack.__main__ import render_k8s
    from simstack.catalog import Stack
    from simstack.render import Renderer

    render_k8s(stack_dir)
    r = Renderer(Stack.load(stack_dir))
    configs = r.sim_configs("physics")
    summary = {
        "stack": f"deploy/{stack}",
        "resource_name": r.base_name(configs),
        "hf_id": hf_id,
        "served_model_name": served,
        "hardware": args.hardware,
        "weight_dtype": weight_dtype,
        "tp": args.tp,
        "engine": r.engine,
        "beta": configs["standalone"]["latency"]["beta"],
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
