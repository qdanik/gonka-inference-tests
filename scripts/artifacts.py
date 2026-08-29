"""Publish the artifacts/*.zip archives to a GitHub Release instead of to git.

Why not git: the archives are ~600 MB of already-compressed test output. Git
stores every version of a zip in full — changing one byte inside an archive
rewrites its whole compressed stream, so there is no usable delta — and GitHub
rejects any single file over 100 MB outright. Release assets have none of that:
2 GiB per file, no repository-size cost, no cost to `git clone`.

What stays in the repository is `artifacts/manifest.json`: the name, size and
sha256 of every archive. A fresh checkout can therefore tell exactly which
archives it is missing and fetch them back byte-identical.

    python -m scripts.artifacts push      # upload changed archives, refresh manifest
    python -m scripts.artifacts pull      # download every archive the manifest lists
    python -m scripts.artifacts status    # compare local dir, manifest and release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
DEFAULT_RELEASE_TAG = "artifacts"
RELEASE_TITLE = "Test artifacts"
RELEASE_NOTES = (
    "Zipped inference/gateway test artifacts.\n\n"
    "Fetch them with `python -m scripts.artifacts pull`. "
    "The repository tracks only `artifacts/manifest.json`."
)
HASH_CHUNK_BYTES = 1 << 20


def run_gh(
    *arguments: str, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a `gh` command.

    Pass `capture=False` for transfers: gh writes its progress bar to stderr, and
    a captured 600 MB upload is indistinguishable from a hang.
    """
    return subprocess.run(
        ["gh", *arguments], check=check, text=True, capture_output=capture
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def local_archives() -> list[Path]:
    return sorted(ARTIFACTS_DIR.glob("*.zip"))


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"release_tag": DEFAULT_RELEASE_TAG, "archives": []}
    return json.loads(MANIFEST_PATH.read_text())


def write_manifest(manifest: dict) -> None:
    manifest["archives"].sort(key=lambda archive: archive["name"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def current_repository() -> str:
    return json.loads(run_gh("repo", "view", "--json", "nameWithOwner").stdout)[
        "nameWithOwner"
    ]


def ensure_release_exists(release_tag: str) -> None:
    if run_gh("release", "view", release_tag, check=False).returncode == 0:
        return
    print(f"creating release {release_tag!r}")
    run_gh(
        "release", "create", release_tag,
        "--title", RELEASE_TITLE, "--notes", RELEASE_NOTES,
    )


def release_asset_names(release_tag: str) -> set[str]:
    listing = run_gh(
        "release", "view", release_tag, "--json", "assets", check=False
    )
    if listing.returncode != 0:
        return set()
    return {asset["name"] for asset in json.loads(listing.stdout)["assets"]}


def resolve_local_conflict(archive_name: str, policy: str) -> bool:
    """Decide what `pull` does when a local archive differs from the manifest.

    A mismatch means one of two things and the command cannot tell them apart:
    the local copy is stale, or it is a *newer* run that was never pushed.
    Returns True to overwrite the local file, False to leave it alone.
    """
    if policy == "overwrite":
        print(f"  {archive_name}: differs from manifest — overwriting")
        return True
    if policy == "fail":
        raise SystemExit(
            f"{archive_name} differs from the manifest. Push it "
            f"(`python -m scripts.artifacts push`) or re-run pull with "
            f"--on-conflict overwrite."
        )
    print(f"  {archive_name}: differs from manifest — keeping local copy (skip)")
    return False


def command_push(args: argparse.Namespace) -> int:
    archives = local_archives()
    if not archives:
        print(f"no *.zip in {ARTIFACTS_DIR}/ — nothing to push")
        return 0

    manifest = load_manifest()
    manifest["release_tag"] = args.tag
    manifest["repository"] = current_repository()
    published = {archive["name"]: archive for archive in manifest["archives"]}
    already_uploaded = release_asset_names(args.tag)

    outdated: list[Path] = []
    for archive_path in archives:
        checksum = sha256_of(archive_path)
        recorded = published.get(archive_path.name)
        is_unchanged = (
            recorded
            and recorded["sha256"] == checksum
            and archive_path.name in already_uploaded
        )
        if is_unchanged:
            print(f"  {archive_path.name}: unchanged, skipping")
            continue
        outdated.append(archive_path)
        published[archive_path.name] = {
            "name": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "sha256": checksum,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    if not outdated:
        print("release is already up to date")
        manifest["archives"] = list(published.values())
        write_manifest(manifest)
        return 0

    total_bytes = sum(archive.stat().st_size for archive in outdated)
    print(f"uploading {len(outdated)} archive(s), {human_size(total_bytes)}")
    if args.dry_run:
        for archive_path in outdated:
            print(f"  would upload {archive_path.name}")
        return 0

    ensure_release_exists(args.tag)
    run_gh(
        "release", "upload", args.tag,
        *[str(archive) for archive in outdated], "--clobber",
        capture=False,
    )
    manifest["archives"] = list(published.values())
    write_manifest(manifest)
    print(f"uploaded. manifest written to {MANIFEST_PATH}")
    return 0


def command_pull(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if not manifest["archives"]:
        print(f"{MANIFEST_PATH} lists no archives — nothing to pull")
        return 0

    release_tag = manifest.get("release_tag", args.tag)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    missing: list[str] = []

    for archive in manifest["archives"]:
        archive_path = ARTIFACTS_DIR / archive["name"]
        if archive_path.exists():
            if sha256_of(archive_path) == archive["sha256"]:
                print(f"  {archive['name']}: present and verified")
                continue
            if not resolve_local_conflict(archive["name"], args.on_conflict):
                continue
        missing.append(archive["name"])

    if not missing:
        print("all archives present")
        return 0
    if args.dry_run:
        for archive_name in missing:
            print(f"  would download {archive_name}")
        return 0

    print(f"downloading {len(missing)} archive(s) from release {release_tag!r}")
    patterns: list[str] = []
    for archive_name in missing:
        patterns += ["--pattern", archive_name]
    run_gh(
        "release", "download", release_tag,
        "--dir", str(ARTIFACTS_DIR), "--clobber", *patterns,
        capture=False,
    )

    for archive in manifest["archives"]:
        if archive["name"] not in missing:
            continue
        downloaded = ARTIFACTS_DIR / archive["name"]
        if sha256_of(downloaded) != archive["sha256"]:
            raise SystemExit(f"checksum mismatch after download: {archive['name']}")
    print("downloaded and verified")
    return 0


def command_status(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    recorded = {archive["name"]: archive for archive in manifest["archives"]}
    on_disk = {archive.name: archive for archive in local_archives()}
    in_release = release_asset_names(manifest.get("release_tag", args.tag))

    for archive_name in sorted(set(recorded) | set(on_disk) | set(in_release)):
        marks = [
            "disk" if archive_name in on_disk else "----",
            "manifest" if archive_name in recorded else "--------",
            "release" if archive_name in in_release else "-------",
        ]
        size = (
            human_size(on_disk[archive_name].stat().st_size)
            if archive_name in on_disk
            else human_size(recorded[archive_name]["size_bytes"])
            if archive_name in recorded
            else "?"
        )
        print(f"  {' '.join(marks)}  {size:>9}  {archive_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.artifacts", description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG, help="release tag holding the archives")
    subcommands = parser.add_subparsers(dest="command", required=True)

    push_parser = subcommands.add_parser("push", help="upload changed archives to the release")
    push_parser.add_argument("--dry-run", action="store_true", help="list what would upload")
    push_parser.set_defaults(handler=command_push)

    pull_parser = subcommands.add_parser("pull", help="download archives listed in the manifest")
    pull_parser.add_argument("--dry-run", action="store_true", help="list what would download")
    pull_parser.add_argument(
        "--on-conflict", choices=("skip", "overwrite", "fail"), default="skip",
        help="what to do when a local archive differs from the manifest",
    )
    pull_parser.set_defaults(handler=command_pull)

    status_parser = subcommands.add_parser("status", help="compare disk, manifest and release")
    status_parser.set_defaults(handler=command_status)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except subprocess.CalledProcessError as failure:
        details = (failure.stderr or "").strip() or (failure.stdout or "").strip()
        print(details or f"gh exited with {failure.returncode}", file=sys.stderr)
        return failure.returncode


if __name__ == "__main__":
    raise SystemExit(main())
