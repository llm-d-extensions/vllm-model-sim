"""simstack CLI: render, check, resolve and promote simulator stacks.

    python -m simstack render deploy/<stack> [...]      # re-render deploy/<stack>/k8s/
    python -m simstack render --all
    python -m simstack render deploy/<stack> --eval physics --out <dir>
    python -m simstack check                            # fail if any k8s/ is stale
    python -m simstack resolve deploy/<stack> [--variant physics]
    python -m simstack promote deploy/<stack> --tuned <tuned-sim-config.json>
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from simstack.catalog import (
    MATCH_KEYS,
    REPO_ROOT,
    VARIANT_TYPES,
    Stack,
    calibration_path,
)
from simstack.render import Renderer, write_files


def all_stacks(root: Path = REPO_ROOT) -> list[Path]:
    return sorted(p.parent for p in (root / "deploy").glob("*/stack.yaml"))


def render_k8s(stack_dir: Path, root: Path = REPO_ROOT) -> list[str]:
    """Write deploy/<stack>/k8s/, removing files no longer rendered."""
    files = Renderer(Stack.load(stack_dir), root).render_deploy()
    out = Path(stack_dir) / "k8s"
    if out.exists():
        for p in out.iterdir():
            if p.is_file() and p.name not in files:
                p.unlink()
    write_files(files, out)
    return sorted(files)


def stale_files(stack_dir: Path, root: Path = REPO_ROOT) -> list[str]:
    """Files in deploy/<stack>/k8s/ that differ from a fresh render."""
    files = Renderer(Stack.load(stack_dir), root).render_deploy()
    out = Path(stack_dir) / "k8s"
    stale = []
    for name, text in files.items():
        p = out / name
        if not p.exists() or p.read_text() != text:
            stale.append(str(p))
    if out.exists():
        stale += [str(p) for p in out.iterdir()
                  if p.is_file() and p.name not in files]
    return sorted(stale)


def dump_calibrations(records: list[dict]) -> str:
    """Records as YAML, with `match` and `latency` kept on one line each."""
    out = []
    for r in records:
        lines = []
        for key, value in r.items():
            if key in ("match", "latency"):
                flow = yaml.safe_dump(value, default_flow_style=True,
                                      sort_keys=False, width=1000).strip()
                text = f"{key}: {flow}"
            else:
                text = yaml.safe_dump({key: value}, sort_keys=False,
                                      width=80).rstrip()
            text = text.replace("\n", "\n  ")
            lines.append(("- " if not lines else "  ") + text)
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def promote(stack_dir: Path, tuned_path: Path, root: Path = REPO_ROOT) -> dict:
    """Write a tuned beta into the stack's exact-match calibration record."""
    stack = Stack.load(stack_dir)
    with open(tuned_path) as f:
        beta = [float(b) for b in json.load(f)["latency"]["beta"]]
    match = {"topology": stack.topology, "tp": stack.tp}
    path = calibration_path(stack.model, stack.hardware, root)
    header, records = "", []
    if path.exists():
        text = path.read_text()
        header = _header(text)
        records = yaml.safe_load(text) or []
    try:
        source = str(Path(tuned_path).resolve().relative_to(root))
    except ValueError:
        source = str(tuned_path)
    record = {"match": match, "latency": {"type": "physics", "beta": beta},
              "method": "fitted", "source": source}
    for i, r in enumerate(records):
        if r.get("match") == match and r["latency"].get("type") == "physics":
            records[i] = record
            break
    else:
        records.insert(0, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dump_calibrations(records)
    path.write_text(header + ("\n" if header else "") + body)
    return record


def _header(text: str) -> str:
    """The file's leading comment block, kept verbatim across promotes."""
    lines = []
    for line in text.splitlines(keepends=True):
        if not (line.startswith("#") or not line.strip()):
            break
        lines.append(line)
    return "".join(lines).rstrip() + "\n" if lines else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="simstack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("render", help="Render deploy/<stack>/k8s/ or an eval stack.")
    rp.add_argument("stacks", nargs="*", type=Path)
    rp.add_argument("--all", action="store_true", help="Render every deploy/*/.")
    rp.add_argument("--eval", dest="variant", choices=sorted(VARIANT_TYPES),
                    help="Render the real+sim eval stack for this latency variant.")
    rp.add_argument("--out", type=Path, help="Output dir for --eval.")

    sub.add_parser("check", help="Fail if any committed k8s/ render is stale.")

    sp = sub.add_parser("resolve", help="Show which calibration each sim pod uses.")
    sp.add_argument("stack", type=Path)
    sp.add_argument("--variant", default="physics", choices=sorted(VARIANT_TYPES))

    pp = sub.add_parser("promote", help="Write a tuned beta into calibrations/.")
    pp.add_argument("stack", type=Path)
    pp.add_argument("--tuned", type=Path, required=True,
                    help="tuned-sim-config.json written by `eval.sh tune`.")

    args = ap.parse_args(argv)

    if args.cmd == "render":
        stacks = all_stacks() if args.all else args.stacks
        if not stacks:
            ap.error("render: give stack dirs or --all")
        if args.variant:
            if len(stacks) != 1 or not args.out:
                ap.error("render --eval needs exactly one stack and --out")
            files = Renderer(Stack.load(stacks[0])).render_eval(args.variant)
            write_files(files, args.out)
            print(f"wrote {len(files)} files to {args.out}")
            return 0
        for s in stacks:
            names = render_k8s(s)
            print(f"{s}/k8s: {', '.join(names)}")
        return 0

    if args.cmd == "check":
        stale = [f for s in all_stacks() for f in stale_files(s)]
        if stale:
            print("stale renders (run `python -m simstack render --all`):")
            for f in stale:
                print(f"  {f}")
            return 1
        print(f"{len(all_stacks())} stacks up to date")
        return 0

    if args.cmd == "resolve":
        r = Renderer(Stack.load(args.stack))
        for role, rec in r.resolve(args.variant).items():
            query = ",".join(f"{k}={v}" for k, v in r.query(role).items()
                             if k in MATCH_KEYS)
            print(f"{role} ({query}):")
            print("  " + yaml.safe_dump(rec, sort_keys=False, width=88)
                  .rstrip().replace("\n", "\n  "))
        return 0

    if args.cmd == "promote":
        rec = promote(args.stack, args.tuned)
        print(f"promoted beta {rec['latency']['beta']} for {rec['match']}")
        names = render_k8s(args.stack)
        print(f"re-rendered {args.stack}/k8s: {', '.join(names)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
