#!/usr/bin/env python3
"""Approved new disposable guest only: bounded, source-free observation; no repair."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

EXPECTED_ARCHIVE = "3c33a179916ec08a83ca8ccb3c19e7682862d0382a63b2c05b5369a3cb124e33"
EXPECTED_BYTES = 18378050
EXPECTED_MEMBER = "386c3ede5ff1687a5e0fe9d1948faf775561db7c4caba14dcfddc3de5087e9b3"
EXPECTED_MEMBER_BYTES = 37690414


def bounded(path, maximum):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Observation bound exceeded")
    return data


def observe():
    receipt = {"schemaVersion": 1, "scope": "B09 dedicated negative-fixture guest observation", "observedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        boot = bounded("/proc/sys/kernel/random/boot_id", 64).decode().strip()
        if not re.fullmatch(r"[0-9a-f-]{36}", boot):
            raise ValueError("Invalid boot identity")
        receipt["bootId"] = boot
        # Small normal status output is consumed privately, never copied to the receipt.
        status = subprocess.run(["cloud-init", "status", "--long"], capture_output=True, timeout=5)
        if len(status.stdout) + len(status.stderr) > 65536:
            raise ValueError("Cloud-init status exceeded bound")
        match = re.search(rb"(?m)^status:\s*([a-z -]+)\s*$", status.stdout)
        label = match.group(1).decode().strip() if match else "unrecognized"
        receipt["cloudInitStatus"] = label if label in ("error", "running", "done", "not run", "disabled") else "unrecognized"
        receipt["cloudInitExitCode"] = status.returncode
        final = subprocess.run(["systemctl", "show", "cloud-final.service", "--property=ActiveState,SubState,Result", "--no-pager"], capture_output=True, timeout=5)
        if len(final.stdout) > 4096 or final.returncode != 0:
            raise ValueError("Cloud-final state unavailable")
        fields = dict(line.split("=", 1) for line in final.stdout.decode().splitlines() if "=" in line)
        receipt["cloudFinalFailed"] = fields.get("ActiveState") == "failed" and fields.get("Result") == "exit-code"
        state = Path("/var/lib/agefreighter")
        installs = list(state.glob("install.*"))
        if len(installs) != 1 or installs[0].is_symlink() or not installs[0].is_dir():
            raise ValueError("Expected exactly one original installation directory")
        work = installs[0]
        archive = work / "archive.tar.gz"
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Archive identity mismatch")
        data = bounded(archive, EXPECTED_BYTES)
        receipt["archiveSHA256"] = hashlib.sha256(data).hexdigest()
        if receipt["archiveSHA256"] != EXPECTED_ARCHIVE:
            raise ValueError("Archive hash mismatch")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source:
            members = source.getmembers()
            receipt["archiveHasOnlyRetainedRegularMember"] = len(members) == 1 and members[0].name == "agefreighter" and members[0].isfile() and members[0].size == EXPECTED_MEMBER_BYTES
        extracted = work / "agefreighter"
        receipt["retainedMemberExtracted"] = extracted.is_file() and not extracted.is_symlink() and hashlib.sha256(bounded(extracted, EXPECTED_MEMBER_BYTES)).hexdigest() == EXPECTED_MEMBER
        # Shell redirection creates the empty destination before missing-member tar fails.
        missing = work / "agefreighter-tools"
        receipt["missingToolsExtractionLeftEmptyFile"] = missing.is_file() and not missing.is_symlink() and missing.stat().st_size == 0
        receipt["bootstrapCompleteAbsent"] = not os.path.lexists(state / "bootstrap.complete")
        receipt["archiveMarkerAbsent"] = not os.path.lexists(state / "evidence/archive.sha256")
        receipt["versionEvidenceAbsent"] = not os.path.lexists(state / "evidence/version.txt")
        receipt["installedExecutablesAbsent"] = all(not os.path.lexists(Path("/usr/local/bin") / name) for name in ("agefreighter", "agefreighter-tools"))
        # Only bounded fixed-text booleans escape the log; never raw output or environment.
        log = bounded("/var/log/cloud-init-output.log", 2 * 1024 * 1024)
        receipt["missingMemberDiagnosticObserved"] = bool(re.search(rb"agefreighter-tools: Not found in archive", log))
        receipt["tarFailureDiagnosticObserved"] = b"tar: Exiting with failure status due to previous errors" in log
        processes = [p for p in Path("/proc").iterdir() if p.name.isdigit()]
        if len(processes) > 4096:
            raise ValueError("Process observation exceeded bound")
        workers = 0
        for process in processes:
            try:
                executable = os.readlink(process / "exe").removesuffix(" (deleted)")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if Path(executable).name in ("agefreighter", "agefreighter-tools"):
                workers += 1
        receipt["runnerExecutableProcessCount"] = workers
        receipt["terminalPackagingFailureObserved"] = (
            receipt["cloudInitStatus"] == "error" and status.returncode != 0
            and receipt["cloudFinalFailed"] and receipt["archiveSHA256"] == EXPECTED_ARCHIVE
            and workers == 0 and all(receipt[field] for field in (
                "archiveHasOnlyRetainedRegularMember", "retainedMemberExtracted", "missingToolsExtractionLeftEmptyFile",
                "bootstrapCompleteAbsent", "archiveMarkerAbsent", "versionEvidenceAbsent", "installedExecutablesAbsent",
                "missingMemberDiagnosticObserved", "tarFailureDiagnosticObserved")))
    except Exception:
        receipt["terminalPackagingFailureObserved"] = False
        receipt["observationError"] = "Bounded observation unavailable or failed; retain evidence and review without retry or repair"
    return receipt


if __name__ == "__main__":
    result = observe()
    encoded = json.dumps(result, sort_keys=True)
    if len(encoded.encode()) >= 4096:
        result = {"schemaVersion": 1, "terminalPackagingFailureObserved": False, "observationError": "Sanitized result exceeded bound"}
        encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    sys.exit(0 if result["terminalPackagingFailureObserved"] else 2)
