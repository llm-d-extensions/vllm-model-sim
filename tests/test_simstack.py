import json
import shutil
from pathlib import Path

import pytest
import yaml

from simstack.__main__ import all_stacks, promote, stale_files
from simstack.catalog import REPO_ROOT, Stack, UNIT_BETA, resolve
from simstack.render import Renderer

STACKS = all_stacks()


def _docs(text):
    return [d for d in yaml.safe_load_all(text) if d]


def _one(files, name):
    return _docs(files[name])[0]


def _container(dep, name="vllm"):
    spec = dep["spec"]["template"]["spec"]
    return next(c for c in spec["containers"] if c["name"] == name)


def _cmdline(c):
    return " ".join(c["command"] + c.get("args", []))


# ---- resolution --------------------------------------------------------------

RECORDS = [
    {"match": {"topology": "pd"}, "latency": {"type": "physics", "beta": [1, 0, 0]}},
    {"match": {"topology": "pd", "tp": 2},
     "latency": {"type": "physics", "beta": [2, 0, 0]}},
    {"latency": {"type": "linear", "base_ms": 5.0}},
]


def test_resolve_most_specific_wins():
    rec = resolve(RECORDS, "physics", {"topology": "pd", "tp": 2, "role": "decode"})
    assert rec["latency"]["beta"] == [2, 0, 0]
    rec = resolve(RECORDS, "physics", {"topology": "pd", "tp": 4, "role": "decode"})
    assert rec["latency"]["beta"] == [1, 0, 0]


def test_resolve_falls_back_to_unit_betas():
    rec = resolve(RECORDS, "physics", {"topology": "standalone", "tp": 1})
    assert rec["latency"]["beta"] == UNIT_BETA
    assert rec["method"] == "assumed"


def test_resolve_wildcard_record():
    rec = resolve(RECORDS, "linear", {"topology": "standalone", "tp": 8})
    assert rec["latency"]["base_ms"] == 5.0


def test_resolve_tie_is_an_error():
    records = [{"match": {"tp": 1}, "latency": {"type": "physics"}},
               {"match": {"topology": "pd"}, "latency": {"type": "physics"}}]
    with pytest.raises(LookupError, match="ambiguous"):
        resolve(records, "physics", {"topology": "pd", "tp": 1})


# ---- committed renders ---------------------------------------------------------


def test_stacks_exist():
    assert STACKS


@pytest.mark.parametrize("stack_dir", STACKS, ids=lambda p: p.name)
def test_committed_k8s_is_up_to_date(stack_dir):
    assert stale_files(stack_dir) == [], (
        "run `python -m simstack render --all` and commit the result")


@pytest.mark.parametrize("stack_dir", STACKS, ids=lambda p: p.name)
def test_every_stack_has_a_sweep(stack_dir):
    assert (stack_dir / "sweep.yaml").exists()


# ---- rendered manifest invariants -----------------------------------------------


@pytest.mark.parametrize("variant", ["physics", "physics-beta-1.0", "flat"])
@pytest.mark.parametrize("stack_dir", STACKS, ids=lambda p: p.name)
def test_eval_stack_invariants(stack_dir, variant):
    r = Renderer(Stack.load(stack_dir))
    files = r.render_eval(variant)
    for name, text in files.items():
        if name.endswith(".yaml"):
            assert all(d.get("kind") for d in _docs(text)), name

    deps = ({"real": ["real-deployment.yaml"], "sim": ["sim-deployment.yaml"]}
            if r.stack.topology == "standalone" else
            {"real": ["real-prefill-deployment.yaml", "real-decode-deployment.yaml"],
             "sim": ["sim-prefill-deployment.yaml", "sim-decode-deployment.yaml"]})

    real_cmds = []
    for f in deps["real"]:
        c = _container(_one(files, f))
        assert c["resources"]["limits"]["nvidia.com/gpu"] == r.stack.tp
        cmd = _cmdline(c)
        assert f"--tensor-parallel-size {r.stack.tp}" in cmd
        assert f"--served-model-name {r.meta['served_model_name']}" in cmd
        real_cmds.append(cmd)

    # Real and sim pods share the engine settings that shape scheduling.
    for f, real_cmd in zip(deps["sim"], real_cmds):
        sim_cmd = _cmdline(_container(_one(files, f)))
        assert f"--served-model-name={r.meta['served_model_name']}" in sim_cmd
        for flag, value in r._engine_flags():
            assert f"{flag} {value}" in real_cmd
            assert f"{flag}={value}" in sim_cmd

    # Services select the deployments eval.sh port-forwards to.
    dep_names = {_one(files, f)["metadata"]["name"]
                 for side in deps.values() for f in side}
    for svc in ("real-service.yaml", "sim-service.yaml"):
        assert _one(files, svc)["spec"]["selector"]["app"] in dep_names

    # The ConfigMap the sim mounts exists and carries the tuned role's config.
    cms = {d["metadata"]["name"]: d for d in _docs(files["configmap.yaml"])}
    tuned = _one(files, deps["sim"][-1])
    vol = next(v for v in tuned["spec"]["template"]["spec"]["volumes"]
               if v["name"] == "model-config")
    cfg = json.loads(cms[vol["configMap"]["name"]]["data"]["config.json"])
    assert cfg == json.loads(files["sim-config.json"])
    assert cfg["architectures"] == ["SimulatedForCausalLM"]
    assert cfg["hidden_size"] == r.hf_config["hidden_size"]
    if variant == "physics-beta-1.0":
        assert cfg["latency"]["beta"] == UNIT_BETA

    # The in-cluster benchmark Job targets the entry-point services.
    job = _one(files, "benchmark-job.yaml")
    script = " ".join(job["spec"]["template"]["spec"]["containers"][0]["command"])
    for svc in ("real-service.yaml", "sim-service.yaml"):
        assert f"{_one(files, svc)['metadata']['name']}:8000" in script


def test_tp_divides_omp_binding_per_rank():
    stack = next(st for st in map(Stack.load, STACKS)
                 if st.tp == 2 and st.topology == "pd")
    files = Renderer(stack).render_deploy()
    env = _container(_one(files, "decode-deployment.yaml"))["env"]
    bind = next(e["value"] for e in env if e["name"] == "VLLM_CPU_OMP_THREADS_BIND")
    assert bind == "0-3|4-7"


# ---- promote --------------------------------------------------------------------


def test_promote_updates_calibration_and_keeps_header(tmp_path):
    for d in ("catalog", "profiles", "calibrations", "deploy"):
        shutil.copytree(REPO_ROOT / d, tmp_path / d)
    stack_dir = tmp_path / "deploy" / "qwen3-32b-h100-sxm5-standalone"
    tuned = tmp_path / "tuned-sim-config.json"
    tuned.write_text(json.dumps({"latency": {"type": "physics",
                                             "beta": [1.5, 1.25, 3.0]}}))
    promote(stack_dir, tuned, root=tmp_path)

    cal = tmp_path / "calibrations" / "qwen3-32b" / "h100-sxm5.yaml"
    assert cal.read_text().startswith("# Latency calibrations")
    stack = Stack.load(stack_dir)
    r = Renderer(stack, root=tmp_path)
    rec = r.resolve("physics")["standalone"]
    assert rec["latency"]["beta"] == [1.5, 1.25, 3.0]
    assert rec["method"] == "fitted"
    assert rec["source"] == "tuned-sim-config.json"
    # Other records survive the rewrite.
    pd = Renderer(Stack.load(tmp_path / "deploy" / "qwen3-32b-h100-sxm5-pd"),
                  root=tmp_path)
    assert pd.resolve("physics")["decode"]["latency"]["beta"] == [
        3.119229, 1.182904, 0.146612]
    assert pd.resolve("flat")["decode"]["latency"]["type"] == "linear"
