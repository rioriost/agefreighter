#!/usr/bin/env python3
"""Local-only negative packaging fixture; never downloads, executes or uploads it."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

PIN = {
    "commit": "d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7",
    "sha256": "2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6",
    "bytes": 37197546,
}
LIMIT = 128 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect_source(manifest_path, pin=PIN):
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 16384:
        raise ValueError("Source manifest must be a bounded regular file, not a symlink")
    manifest = json.loads(manifest_path.read_text())
    expected = dict(pin, schemaVersion=1, platform="linux-amd64", version="2.4.0-dev." + pin["commit"][:12])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Source manifest does not match the reviewed original")
    name = manifest.get("archive", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar\.gz", name):
        raise ValueError("Source archive must be a same-directory filename")
    path = manifest_path.parent / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != pin["bytes"] or pin["bytes"] > LIMIT:
        raise ValueError("Source archive differs from reviewed size or is not a regular file")
    data = path.read_bytes()
    if len(data) != pin["bytes"] or digest(data) != pin["sha256"]:
        raise ValueError("Source archive differs from reviewed bytes")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if sorted(m.name for m in members) != ["agefreighter", "agefreighter-tools"]:
            raise ValueError("Source must contain exactly the two expected members")
        if any(not m.isfile() or m.size < 1 or m.size > 2 * LIMIT for m in members):
            raise ValueError("Source executable member is not a bounded regular file")
        binary = archive.extractfile("agefreighter").read()
    return manifest, binary


def prepare(manifest_path, output, pin=PIN):
    original, binary = inspect_source(manifest_path, pin)
    # New explicit directory only. Never replace an existing fixture or archive.
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    name = "b09-negative-missing-tools.tar.gz"
    path = output / name
    with path.open("xb") as file:
        path.chmod(0o600)
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                member = tarfile.TarInfo("agefreighter")
                member.size = len(binary)
                member.mode = 0o755
                member.mtime = 0
                archive.addfile(member, io.BytesIO(binary))
    data = path.read_bytes()
    if len(data) > LIMIT:
        raise ValueError("Derivative exceeds the production artifact bound")
    manifest = {key: original[key] for key in ("schemaVersion", "platform", "version", "commit")}
    manifest.update(sha256=digest(data), bytes=len(data), archive=name)
    provenance = {
        "schemaVersion": 1,
        "purpose": "B09 intentional terminal bootstrap packaging failure on one new disposable VM only",
        "negativeFixture": True,
        "usableRunner": False,
        "sourceCommitMeaning": "Included executable source; this derivative is not an unmodified build or release",
        "original": {key: original[key] for key in ("commit", "sha256", "bytes", "archive")},
        "retainedMember": {"name": "agefreighter", "sha256": digest(binary), "bytes": len(binary)},
        "omittedMember": "agefreighter-tools",
        "construction": "Python USTAR one regular member, mode 0755, uid/gid/mtime 0, empty owner names; gzip level 9, mtime 0, empty filename",
        "derivative": manifest,
        "expectedFailure": "Second tar extraction; set -e terminates before both executable installs, version checks, archive marker and bootstrap.complete",
        "prohibitedUse": "Never upgrade an existing runner, replace a published artifact, or run a source operation with this fixture",
    }
    # Keep the normal production manifest small and compatible; provenance is mandatory separate review.
    for filename, value in (("manifest.json", manifest), ("negative-fixture-provenance.json", provenance)):
        target = output / filename
        with target.open("x") as stream:
            target.chmod(0o600)
            json.dump(value, stream, indent=2)
            stream.write("\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Reviewed original development manifest")
    parser.add_argument("output", type=Path, help="New private output directory; must not exist")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
