"""Record the goldens of the rollback check (tests/test_legacy_profile.py).

Runs the scenarios of tests/legacy_scenarios.py against the package of another checkout - the
commit the rollback profile must reproduce, ae66b8f - and writes their events, per-frame
statuses and tracker states, plus that commit's config defaults, to a JSON file:

    git worktree add ../gc-ae66b8f ae66b8f          # or any clean checkout of the commit
    python tools/record_legacy_goldens.py --repo ../gc-ae66b8f --out tests/data/legacy_events.json

The scenario definitions always come from *this* checkout's tests/ directory; only the
``gradecrossing`` package is imported from ``--repo``. Run it once per intended change of the
pre-port behaviour (never to make a failing rollback check pass).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Record the rollback-profile goldens from another checkout.")
    p.add_argument("--repo", required=True, help="checkout whose gradecrossing package produces the goldens")
    p.add_argument("--out", default=str(ROOT / "tests" / "data" / "legacy_events.json"), help="JSON file to write")
    args = p.parse_args(argv)
    repo = Path(args.repo).resolve()
    if not (repo / "gradecrossing" / "__init__.py").is_file():
        print(f"error: no gradecrossing package under {repo}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(repo))  # first: the package under test comes from --repo
    import gradecrossing  # noqa: E402
    from gradecrossing.config import MonitorConfig  # noqa: E402

    if Path(gradecrossing.__file__).resolve().parents[1] != repo:
        print(f"error: imported gradecrossing from {gradecrossing.__file__}, not {repo}", file=sys.stderr)
        return 2
    import legacy_scenarios  # noqa: E402

    defaults = dataclasses.asdict(MonitorConfig())
    data = {
        "recorded_from": {"repo": repo.name, "commit": _commit(repo)},
        "defaults": {k: defaults[k] for k in ("stuck", "alerts", "detector")},
        "scenarios": legacy_scenarios.record_all(None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n_events = sum(len(s["events"]) for s in data["scenarios"]["monitor"].values())
    print(f"wrote {out} ({data['recorded_from']['commit']}): {n_events} events, "
          f"{len(data['scenarios']['tracker'])} tracker scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
