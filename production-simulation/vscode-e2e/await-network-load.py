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

WORKFLOW = "8a9ae99e-c621-4a94-afd1-a30ff210a201"
TARGET = "afpg-8a9ae99ec6214a94afd1.postgres.database.azure.com"
GRAPH = "neo4j526_network_recovery_p1_r1"
UUID = r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}"


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


def candidates(root):
    found = []
    for path in root.glob("*/state.json"):
        state = json.loads(path.read_text())
        if state.get("action") in ("migrate-source", "migrate-csv", "resume-migration"):
            found.append((path.parent, state))
    assert len(found) <= 1, "More than one migration operation requires manual review"
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--observer-sha256", required=True)
    args = parser.parse_args()
    assert __debug__ and os.geteuid() == 0 and re.fullmatch(UUID, args.boot)
    observer_path = Path(__file__).with_name("observe-recovery-guest.py")
    assert hashlib.sha256(observer_path.read_bytes()).hexdigest() == args.observer_sha256
    spec = importlib.util.spec_from_file_location("observer", observer_path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    root = Path("/var/lib/agefreighter/workflows") / WORKFLOW
    assert root.is_dir() and not candidates(root), "Never attach to a pre-existing load"
    started = dt.datetime.now(dt.timezone.utc)
    deadline = observer.timestamp(args.deadline)
    assert started < deadline <= started + dt.timedelta(minutes=16)
    observer.seal(root / "qualification-network-armed.json", {
        "workflow": WORKFLOW, "boot": args.boot, "startedAt": started.isoformat(),
        "deadline": args.deadline, "observerSHA256": args.observer_sha256,
        "target": TARGET, "graph": GRAPH, "startsMigration": False})
    while dt.datetime.now(dt.timezone.utc) < deadline:
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == args.boot
        found = candidates(root)
        if found:
            directory, state = found[0]
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
        print(json.dumps({"stopped": True, "errorType": type(error).__name__}), flush=True)
        raise SystemExit(1)
