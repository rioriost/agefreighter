#!/usr/bin/env python3
"""NOEXEC builder for up to two pre-grant and three manual post-grant reads.

Input must be an actual approved binding and actual nonsecret VM identity receipt.
No placeholder runtime IDs/times are supplied by this tool. No cloud launcher.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBSERVER = HERE / "b06-observe-access-v2.py"
spec = importlib.util.spec_from_file_location("b06_v2_observer", OBSERVER)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def durable(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def body(config, observer_sha):
    observer.validate(config)
    code = OBSERVER.read_bytes()
    observer.require(sha(code) == observer_sha)
    config_raw = observer.encoded(config)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")
    # Initialization fallback deliberately makes no claim about request counts.
    # The observer owns the complete runtime/transport exception boundary.
    source = "\n".join([
        "set -eu", "exec /usr/bin/python3 - <<'AF_B06_V2'",
        "import base64, hashlib, json, sys, types",
        "try:",
        f"    code = base64.b64decode({b64(code)!r})",
        f"    raw = base64.b64decode({b64(config_raw)!r})",
        f"    assert hashlib.sha256(code).hexdigest() == {observer_sha!r}",
        f"    assert hashlib.sha256(raw).hexdigest() == {sha(config_raw)!r}",
        "    module = types.ModuleType('b06_v2_pinned_observer')",
        "    exec(compile(code, 'b06_v2_pinned_observer', 'exec'), module.__dict__)",
        f"    status = module.run_once(json.loads(raw), {sha(config_raw)!r}, {observer_sha!r})",
        "except Exception:",
        "    print(json.dumps({'schemaVersion':2,'event':'result','classification':'inconclusive','stage':'initialization-or-output','reason':'runtime-unavailable','requestCountsKnown':False,'guiAssessment':False}), flush=True)",
        "    status = 2",
        "sys.exit(status)", "AF_B06_V2", ""])
    return {"location": "japaneast", "tags": {"application": "agefreighter", "workflow": config["trialId"],
            "purpose": "b06-cosmos-access-observation"},
            "properties": {"source": {"script": source}, "timeoutInSeconds": 60, "asyncExecution": False}}, config_raw


def prepare(binding_path, binding_sha, identity_path, identity_sha, observer_sha, output):
    for path in (binding_path, identity_path):
        observer.require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 16384)
    raw, identity_raw = binding_path.read_bytes(), identity_path.read_bytes()
    observer.require(len(raw) <= 16384 and len(identity_raw) <= 16384 and sha(raw) == binding_sha and sha(identity_raw) == identity_sha)
    binding, receipt = json.loads(raw), json.loads(identity_raw)
    observer.require(isinstance(binding, dict) and set(binding) == observer.FIELDS - {"phase", "attempt"})
    observer.require(set(receipt) == {"observedAtUTC", "responseSHA256", "id", "identity"})
    identity = receipt["identity"]
    observer.require(set(identity) == {"type", "principalId", "tenantId"} and identity["type"] == "SystemAssigned")
    observer.require(receipt["id"] == binding["vmId"] and identity["principalId"] == binding["principalId"] and identity["tenantId"] == binding["tenantId"])
    observer.require(receipt["observedAtUTC"] == binding["vmIdentityObservedAtUTC"] and receipt["responseSHA256"] == binding["vmIdentityReceiptSHA256"])
    configs = [{**binding, "phase": phase, "attempt": attempt} for phase, attempt in
               (("before-grant", 1), ("before-grant", 2), ("after-grant", 1), ("after-grant", 2), ("after-grant", 3))]
    # Fully validate all bodies before creating the output directory.
    prepared = [(config, *body(config, observer_sha)) for config in configs]
    output.mkdir(mode=0o700)
    observer.require(output.resolve() == output and output.stat().st_mode & 0o777 == 0o700)
    durable(output / "binding.json", raw)
    durable(output / "vm-identity-receipt.json", identity_raw)
    artifacts = []
    for config, request, config_raw in prepared:
        phase = config["phase"].split("-")[0]
        name = f"af-b06-v2-{phase}-{config['attempt']:02d}"
        request_raw = (json.dumps(request, indent=2) + "\n").encode()
        durable(output / (name + "-body.json"), request_raw)
        durable(output / (name + "-config.json"), config_raw)
        artifacts.append({"id": binding["vmId"] + "/runCommands/" + name, "body": name + "-body.json",
                          "bodySHA256": sha(request_raw), "config": name + "-config.json", "configSHA256": sha(config_raw)})
    result = {"schemaVersion": 2, "preparedOnly": True, "executed": False,
              "workflow": binding["trialId"], "vmId": binding["vmId"], "observerSHA256": observer_sha,
              "bindingFileSHA256": binding_sha, "identityFileSHA256": identity_sha,
              "maximumBefore": 2, "maximumAfter": 3, "maximumSecondsPerCall": 60,
              "maximumIMDSRequestsPerCall": 2, "maximumCosmosQueriesPerCall": 1,
              "maximumOutputBytesPerCall": 4096, "automaticRetries": 0,
              "notAfterUTC": binding["notAfterUTC"], "artifacts": artifacts,
              "dispatchCondition": "Fresh exact approval, guard, VM identity, capacity/idle and per-ID durable once intent required. Second pre-grant attempt requires manual diagnostic review; no automatic retry or replay. Stop at first recognized pre-grant denial and first post-grant success."}
    durable(output / "preparation.json", (json.dumps(result, indent=2) + "\n").encode())
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--binding-sha256", required=True)
    p.add_argument("--vm-identity-receipt", type=Path, required=True)
    p.add_argument("--identity-sha256", required=True)
    p.add_argument("--observer-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(prepare(args.binding, args.binding_sha256, args.vm_identity_receipt, args.identity_sha256, args.observer_sha256, args.output)))
        return 0
    except Exception:
        print('{"preparedOnly":true,"executed":false,"error":"binding-or-preparation-refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
