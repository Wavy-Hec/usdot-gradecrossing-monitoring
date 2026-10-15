#!/usr/bin/env python
"""Download the team benchmark clips into a local cache and verify them.

The clips listed in ``benchmarks/team_clips/clips.yaml`` live on the team's Google Drive.
This tool fetches each one with ``urllib`` from Drive's direct-download endpoint, checks its
size and sha256 against the registry, and only then moves it into place, so the cache never
holds a truncated or substituted file. Cached files that still verify are not downloaded
again. Nothing beyond the standard library and PyYAML is needed (no gdown).

Examples::

    python tools/fetch_clips.py                          # all clips into the default cache
    python tools/fetch_clips.py --clips clip07,clip08 --cache-dir data/clips
    python tools/fetch_clips.py --verify-only            # check the cache, download nothing
    python tools/fetch_clips.py --decode-check           # also count decodable frames (OpenCV)

The default cache is ``$GRADECROSSING_CLIPS``, else ``~/.cache/gradecrossing/team_clips``:
outside the repository, because the videos must never be committed. Exit status: 0 when
every requested clip is present and verified, 1 otherwise, 2 for a bad registry or
arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "benchmarks" / "team_clips" / "clips.yaml"
DEFAULT_URL = "https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
REQUIRED_KEYS = ("id", "drive_id", "file", "size_bytes", "sha256")
CHUNK = 1 << 20


class RegistryError(ValueError):
    """The clip registry is missing, unreadable or incomplete."""


class DownloadError(RuntimeError):
    """A clip could not be downloaded or did not verify."""


def default_cache_dir() -> Path:
    env = os.environ.get("GRADECROSSING_CLIPS")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "gradecrossing" / "team_clips"


def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    """Read clips.yaml; ``clips`` becomes an ordered ``{id: entry}`` dict. Raises RegistryError."""
    import yaml  # local import so --help works without PyYAML

    path = str(path or DEFAULT_REGISTRY)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"cannot read clip registry {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list) or not data["clips"]:
        raise RegistryError(f"{path}: expected a mapping with a non-empty 'clips' list")
    clips: Dict[str, Dict[str, Any]] = {}
    for i, entry in enumerate(data["clips"]):
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: clips[{i}] is not a mapping")
        missing = [k for k in REQUIRED_KEYS if entry.get(k) in (None, "")]
        if missing:
            raise RegistryError(f"{path}: clips[{i}] ({entry.get('id')}) is missing {missing}")
        sha = str(entry["sha256"]).lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise RegistryError(f"{path}: {entry['id']}: sha256 must be 64 hex digits")
        if str(entry["id"]) in clips:
            raise RegistryError(f"{path}: duplicate clip id {entry['id']}")
        entry = dict(entry, id=str(entry["id"]), sha256=sha, size_bytes=int(entry["size_bytes"]))
        clips[entry["id"]] = entry
    data = dict(data)
    data["clips"] = clips
    data["path"] = path
    data["base_dir"] = str(Path(path).resolve().parent)
    return data


def select_clips(registry: Dict[str, Any], names: Optional[str]) -> List[Dict[str, Any]]:
    """Entries for a comma-separated id list (None/"" = all, registry order). Raises RegistryError."""
    clips = registry["clips"]
    if not names:
        return list(clips.values())
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in clips]
    if unknown:
        raise RegistryError(f"unknown clip id(s) {unknown}; registry has {list(clips)}")
    return [clips[n] for n in wanted]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, entry: Dict[str, Any]) -> Optional[str]:
    """None when ``path`` is the registered file, else the reason it is not."""
    if not path.is_file():
        return "not downloaded"
    size = path.stat().st_size
    if size != entry["size_bytes"]:
        return f"size {size} != {entry['size_bytes']}"
    digest = file_sha256(path)
    if digest != entry["sha256"]:
        return f"sha256 {digest[:12]}... != {entry['sha256'][:12]}..."
    return None


def download(url: str, dest: Path, entry: Dict[str, Any], timeout: float = 60.0) -> None:
    """Stream ``url`` to ``dest`` via ``dest.part``; keep it only if size and sha256 match."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gradecrossing-fetch-clips"})
    h = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = str(resp.headers.get("Content-Type", ""))
            if ctype.lower().startswith("text/html"):
                raise DownloadError("the server sent an HTML page instead of the video (Drive quota "
                                    "exceeded, the file is no longer shared, or a confirmation page)")
            with open(part, "wb") as out:
                for chunk in iter(lambda: resp.read(CHUNK), b""):
                    size += len(chunk)
                    if size > entry["size_bytes"]:
                        raise DownloadError(f"more than the expected {entry['size_bytes']} bytes")
                    h.update(chunk)
                    out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"download failed: {exc}") from exc
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    if size != entry["size_bytes"] or h.hexdigest() != entry["sha256"]:
        part.unlink(missing_ok=True)
        raise DownloadError(f"got {size} bytes with sha256 {h.hexdigest()[:12]}..., expected "
                            f"{entry['size_bytes']} bytes with sha256 {entry['sha256'][:12]}...")
    os.replace(str(part), str(dest))


def decoded_frames(path: Path) -> int:
    """Frames OpenCV actually decodes (containers over-report CAP_PROP_FRAME_COUNT)."""
    import cv2  # local import: only --decode-check needs OpenCV

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DownloadError(f"OpenCV cannot open {path}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def clip_path(cache_dir: Path, entry: Dict[str, Any]) -> Path:
    return Path(cache_dir) / str(entry["file"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and verify the team benchmark clips (benchmarks/team_clips/clips.yaml).")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="clip registry YAML (default: %(default)s)")
    p.add_argument("--cache-dir", default=None,
                   help="where the videos go (default: $GRADECROSSING_CLIPS or ~/.cache/gradecrossing/team_clips)")
    p.add_argument("--clips", default="", help="comma-separated clip ids (default: all)")
    p.add_argument("--verify-only", action="store_true", help="check the cached files, download nothing")
    p.add_argument("--force", action="store_true", help="download again even when the cached file verifies")
    p.add_argument("--decode-check", action="store_true",
                   help="also decode each clip with OpenCV and compare the frame count with the registry")
    p.add_argument("--url-template", default=None,
                   help="download URL with {drive_id} (default: the registry's download_url)")
    p.add_argument("--timeout", type=float, default=60.0, help="socket timeout per read, seconds")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.registry)
        entries = select_clips(registry, args.clips)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cache = Path(args.cache_dir).expanduser() if args.cache_dir else default_cache_dir()
    template = args.url_template or str(registry.get("download_url") or DEFAULT_URL)
    if "{drive_id}" not in template:
        print("error: the URL template needs a {drive_id} placeholder", file=sys.stderr)
        return 2
    print(f"cache: {cache}")
    failed = []
    for entry in entries:
        path = clip_path(cache, entry)
        problem = None if args.force else verify_file(path, entry)
        state = "cached"
        if args.force or problem is not None:
            if args.verify_only:
                print(f"{entry['id']}: {problem}")
                failed.append(entry["id"])
                continue
            mb = entry["size_bytes"] / 1e6
            print(f"{entry['id']}: downloading {mb:.1f} MB ({'forced' if args.force else problem})", flush=True)
            t0 = time.monotonic()
            try:
                download(template.format(drive_id=entry["drive_id"]), path, entry, timeout=args.timeout)
            except DownloadError as exc:
                print(f"{entry['id']}: FAILED: {exc}", file=sys.stderr)
                failed.append(entry["id"])
                continue
            state = f"downloaded in {time.monotonic() - t0:.1f}s"
        line = f"{entry['id']}: ok ({state}, sha256 verified) {path}"
        if args.decode_check:
            try:
                n = decoded_frames(path)
            except (DownloadError, ImportError) as exc:
                print(f"{entry['id']}: FAILED decode check: {exc}", file=sys.stderr)
                failed.append(entry["id"])
                continue
            expected = entry.get("frames")
            line += f", {n} decoded frames" + ("" if expected in (None, n) else f" (registry says {expected})")
            if expected is not None and n != int(expected):
                failed.append(entry["id"])
        print(line)
    if failed:
        print(f"{len(failed)} of {len(entries)} clip(s) not ready: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"{len(entries)} clip(s) ready in {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
