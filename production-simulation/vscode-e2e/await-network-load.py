#!/usr/bin/env python3
"""One-workflow guest qualification watcher; never starts or resumes a load.

Arm only after fresh cloud and idle-guest gates, before private GUI password
entry. Refuses all pre-existing migration operations. Expires after 15 minutes;
the observer has a separate 45-second independent network-restoration timer.
This narrowly pinned trial helper is not shipped in the product.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

# R2 is a new installed-GUI draft. R1 completed without a fault and must never
# be rearmed. Confirm these planned identities against the final target review
# before uploading/arming; the original R1 guest files remain immutable.
WORKFLOW = "b2c7214e-83f5-4613-b378-98d36e0cd97d"
TARGET = "afpg-b2c7214e83f54613b378.postgres.database.azure.com"
GRAPH = "neo4j526_network_recovery_p1_r2"
UUID = r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}"
EMPTY_FAILURE = "068ff3f1-41b5-4685-a3eb-e140543d3def"


def empty_failure(root, proof, digest, now, timestamp):
    """Only the reviewed pre-load authentication failure, with fresh empty DB proof."""
    assert proof.parent.parent == root and re.fullmatch("diagnostic-" + UUID, proof.parent.name)
    assert proof.name == "doctor.json" and not proof.is_symlink()
    assert hashlib.sha256(proof.read_bytes()).hexdigest() == digest
    doc = json.loads(proof.read_text())
    assert doc["command"] == "doctor" and doc.get("errors") == []
    assert 0 <= (now - timestamp(doc["generatedAt"])).total_seconds() <= 900
    checks = {c["id"]: c for c in doc["checks"]}
    assert checks["metadata-schema"]["status"] == "unavailable"
    assert checks["metadata-schema"]["detail"] == "installed=0 supported=21 pending=0; doctor does not migrate"
    assert checks["target-graph"]["status"] == "pass"
    assert checks["target-graph"]["summary"] == 'target graph "' + GRAPH + '" is absent'
    directory = root / EMPTY_FAILURE
    state = json.loads((directory / "state.json").read_text())
    assert state["workflow"] == WORKFLOW and state["operation"] == state["jobId"] == EMPTY_FAILURE
    assert state["action"] == "migrate-source" and state["phase"] == "failed" and state["exitCode"] == 1
    assert timestamp(doc["generatedAt"]) > timestamp(state["finishedAt"])
    assert state["configSha256"] == hashlib.sha256((directory / "job.json").read_bytes()).hexdigest() == "19ce9281467d7969ae733bae303ae27471f961c2ecade0610b2e721cd42038f5"
    assert hashlib.sha256((directory / "load.stderr.log").read_bytes()).hexdigest() == "7cade80c58ef868a3d8b00a76bc73129025d8d798ff35e2dce68d472bc222875"
    assert (directory / "load.json").stat().st_size == 0
    assert not (directory / "secrets.json").exists() and not list(directory.glob("qualification-network-*.json"))
    assert not (root / "active").exists() and not (root / "qualification-network-selected.json").exists()
    return EMPTY_FAILURE


def binding(state, configuration, boot):
    assert state["version"] == 1 and state["workflow"] == WORKFLOW
    assert re.fullmatch(UUID, state["operation"])
    assert state["jobId"] == state["operation"] and state["action"] == "migrate-source"
    assert state["bootId"] == boot and state["phase"] in ("accepted", "running")
    assert configuration["source"]["type"] == "neo4j"
    source = configuration["source"]["neo4j"]
    assert source["uri"] == "neo4j+s://neo4j526.azn526.internal:7687"
    assert source["database"] == "neo4j" and source["sourceId"] == WORKFLOW
    assert configuration["target"]["graph"] == GRAPH


def candidates(root, excluded=None):
    found = []
    for path in root.glob("*/state.json"):
        state = json.loads(path.read_text())
        if state.get("action") in ("migrate-source", "migrate-csv", "resume-migration"):
            if excluded is not None and path.parent.name == excluded:
                assert state["operation"] == state["jobId"] == excluded and state["phase"] == "failed"
                continue
            found.append((path.parent, state))
    assert len(found) <= 1, "More than one migration operation requires manual review"
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--empty-failure-proof")
    parser.add_argument("--proof-sha256")
    args = parser.parse_args()
    assert __debug__ and os.geteuid() == 0 and re.fullmatch(UUID, args.boot)
    observer_path = Path(__file__).with_name("observe-recovery-guest.py")
    assert hashlib.sha256(observer_path.read_bytes()).hexdigest() == args.observer_sha256
    spec = importlib.util.spec_from_file_location("observer", observer_path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    root = Path("/var/lib/agefreighter/workflows") / WORKFLOW
    started = dt.datetime.now(dt.timezone.utc)
    assert bool(args.empty_failure_proof) == bool(args.proof_sha256)
    excluded = empty_failure(root, Path(args.empty_failure_proof), args.proof_sha256, started, observer.timestamp) if args.empty_failure_proof else None
    assert root.is_dir() and not candidates(root, excluded), "Never attach to a pre-existing load"
    deadline = observer.timestamp(args.deadline)
    assert started < deadline <= started + dt.timedelta(minutes=16)
    armed_name = "qualification-network-armed-after-auth.json" if excluded else "qualification-network-armed.json"
    observer.seal(root / armed_name, {
        "workflow": WORKFLOW, "boot": args.boot, "startedAt": started.isoformat(),
        "deadline": args.deadline, "observerSHA256": args.observer_sha256,
        "target": TARGET, "graph": GRAPH, "startsMigration": False,
        "excludedEmptyFailure": excluded, "emptyProofSHA256": args.proof_sha256})
    while dt.datetime.now(dt.timezone.utc) < deadline:
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == args.boot
        found = candidates(root, excluded)
        if found:
            directory, state = found[0]
            assert state["phase"] in ("accepted", "running"), "New operation stopped before watcher binding"
            # The worker persists state before its config/secrets; wait only for
            # those create-only files, not for a different or replacement job.
            if not all((directory / name).is_file() for name in ("job.json", "secrets.json")):
                time.sleep(1)
                continue
            raw = (directory / "job.json").read_bytes()
            configuration = json.loads(raw)
            binding(state, configuration, args.boot)
            assert directory.name == state["operation"]
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == state["configSha256"]
            # Inspect only in memory; never print or export this credential.
            dsn = json.loads((directory / "secrets.json").read_text())["AGEFREIGHTER_TARGET_DSN"]
            uri = urlsplit(dsn)
            assert uri.hostname == TARGET and uri.port == 5432 and uri.path == "/agefreighter"
            del dsn, uri
            observer.seal(root / "qualification-network-selected.json", {
                "workflow": WORKFLOW, "operation": state["operation"],
                "job": state["jobId"], "configSHA256": digest, "boot": args.boot})
            sys.argv = [str(observer_path), "--workflow", WORKFLOW,
                        "--operation", state["operation"], "--job", state["jobId"],
                        "--config-sha256", digest, "--boot", args.boot,
                        "--deadline", args.deadline, "--watch-seconds", "600",
                        "--source-kind", "neo4j", "--network-source-ip", "10.246.5.4"]
            observer.main()
            return
        time.sleep(1)
    print(json.dumps({"expired": True, "migrationStarted": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Retain source locations, never exception text, arguments or locals.
        import traceback
        frame = traceback.extract_tb(error.__traceback__)[-1]
        print(json.dumps({"stopped": True, "errorType": type(error).__name__,
                          "line": frame.lineno, "function": frame.name}), flush=True)
        raise SystemExit(1)
