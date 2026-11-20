"""Load the hardware/model catalog, engine profiles and calibration records."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

UNIT_BETA = [1.0, 1.0, 0.0]
MATCH_KEYS = ("topology", "tp", "role")
# Variant name (CLI / results dir) -> latency type of the records it resolves.
VARIANT_TYPES = {"physics": "physics", "physics-beta-1.0": "physics", "flat": "linear"}


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Stack:
    """A deployable combination, read from deploy/<name>/stack.yaml."""

    name: str
    dir: Path
    model: str
    hardware: str
    topology: str  # "standalone" or "pd"
    tp: int
    profile: str

    @classmethod
    def load(cls, stack_dir) -> "Stack":
        d = Path(stack_dir).resolve()
        spec = _load_yaml(d / "stack.yaml")
        topology = spec["topology"]
        if topology not in ("standalone", "pd"):
            raise ValueError(f"{d}: unknown topology {topology!r}")
        tp = spec.get("tp", 1)
        if not isinstance(tp, int) or tp < 1:
            raise ValueError(f"{d}: tp must be a positive integer, got {tp!r}")
        return cls(
            name=d.name,
            dir=d,
            model=spec["model"],
            hardware=spec["hardware"],
            topology=topology,
            tp=tp,
            profile=spec.get("profile", "default"),
        )


def load_hardware(slug: str, root: Path = REPO_ROOT) -> dict:
    return _load_yaml(root / "catalog" / "hardware" / f"{slug}.yaml")


def load_model(slug: str, root: Path = REPO_ROOT) -> tuple[dict, dict]:
    """Return (model.yaml metadata, HF config.json) for a catalog model."""
    d = root / "catalog" / "models" / slug
    meta = _load_yaml(d / "model.yaml")
    with open(d / "config.json") as f:
        hf_config = json.load(f)
    return meta, hf_config


def load_profile(name: str, root: Path = REPO_ROOT) -> dict:
    return _load_yaml(root / "profiles" / f"{name}.yaml")


def calibration_path(model: str, hardware: str, root: Path = REPO_ROOT) -> Path:
    return root / "calibrations" / model / f"{hardware}.yaml"


def load_calibrations(model: str, hardware: str, root: Path = REPO_ROOT) -> list:
    path = calibration_path(model, hardware, root)
    if not path.exists():
        return []
    records = _load_yaml(path) or []
    for r in records:
        unknown = set(r.get("match") or {}) - set(MATCH_KEYS)
        if unknown:
            raise ValueError(f"{path}: unknown match keys {sorted(unknown)}")
    return records


def resolve(records: list, latency_type: str, query: dict) -> dict:
    """Pick the most specific record of `latency_type` whose match fits `query`.

    Returns the record. With no match, physics falls back to a synthetic
    unit-beta record; other types raise.
    """
    candidates = []
    for r in records:
        if r["latency"].get("type") != latency_type:
            continue
        match = r.get("match") or {}
        if all(query.get(k) == v for k, v in match.items()):
            candidates.append(r)
    if not candidates:
        if latency_type == "physics":
            return {"latency": {"type": "physics", "beta": UNIT_BETA},
                    "method": "assumed", "note": "no calibration; unit betas"}
        raise LookupError(f"no {latency_type!r} calibration matches {query}")
    best = max(len(r.get("match") or {}) for r in candidates)
    top = [r for r in candidates if len(r.get("match") or {}) == best]
    if len(top) > 1:
        raise LookupError(
            f"ambiguous calibrations for {query}: "
            f"{[r.get('match') for r in top]}"
        )
    return top[0]
