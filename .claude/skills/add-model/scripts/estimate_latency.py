#!/usr/bin/env python3
"""Print the step times a sim-config.json implies, as a sanity check.

Uses the plugin's own PhysicsLatencyModel (no vLLM import needed), so the
numbers are exactly what the simulator will sleep for.

  python .claude/skills/add-model/scripts/estimate_latency.py \
      models/<slug>/deployments/<hw>/standalone/k8s/sim-config.json [--tp N]
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, os.path.join(REPO, "src"))

from vllm_simulated.physics_latency import PhysicsLatencyModel  # noqa: E402


def shape(pf_tokens=0, pf_seqs=0, dc_seqs=0, ctx=0):
    # A prefill of pf_tokens from scratch plus dc_seqs decodes at context ctx.
    return SimpleNamespace(
        num_prefill_tokens=pf_tokens,
        num_prefill_seqs=pf_seqs,
        num_decode_seqs=dc_seqs,
        sum_decode_context_len=dc_seqs * ctx,
        sum_context_len=dc_seqs * ctx + pf_tokens,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_config")
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args(argv)

    with open(args.sim_config) as f:
        cfg = json.load(f)
    lat = dict(cfg["latency"], tp=args.tp)
    if lat.get("type") != "physics":
        sys.exit("only physics configs are estimated")
    m = PhysicsLatencyModel.from_dict(lat, hf_config=SimpleNamespace(**cfg))

    print(f"beta={lat['beta']} tp={args.tp} hardware={lat['hardware']}")
    print("prefill (single request, ms):")
    for n in (128, 1024, 4096):
        print(f"  {n:>5} tokens  {m.step_time_ms(shape(pf_tokens=n, pf_seqs=1)):8.2f}")
    print("decode step = ITL (ms):")
    for b in (1, 16, 64):
        for ctx in (1024, 4096):
            t = m.step_time_ms(shape(dc_seqs=b, ctx=ctx))
            print(f"  batch {b:>3} ctx {ctx:>5}  {t:8.2f}")


if __name__ == "__main__":
    main()
