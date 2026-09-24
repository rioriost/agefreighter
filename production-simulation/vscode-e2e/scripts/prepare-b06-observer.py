#!/usr/bin/env python3
"""Local-only sealed B06 request preparation. No Azure launcher or guest execution.

The root operator supplies a nonsecret extract of an actual retained ARM VM GET;
its response hash is a provenance reference, not independent proof of that GET.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OBSERVER = HERE.parent / "observe-b06-cosmos-access.py"
OBSERVER_SHA = "004ff881b502145832b4bd842f19bc926dc48d2150dd8365db244a9917a15b5e"
WORKFLOW = "0f83d520-cd03-4792-8e38-3f146abfde39"
BASE = "/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-af-vscode-p1-20260905-a/providers/"
VM = BASE + "Microsoft.Compute/virtualMachines/af-0f83d520cd0347928e38"
ACCOUNT = BASE + "Microsoft.DocumentDB/databaseAccounts/afcosmosp120260907"
APPROVED = "2026-09-24T00:18:14Z"
ABSOLUTE_DEADLINE = "2026-09-24T04:18:14Z"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def require(value):
    if not value:
        raise ValueError("Invalid sealed trial binding")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() == dt.timedelta(0))
    return parsed


def bindings(receipt, first_intent, now=None):
    require(set(receipt) == {"observedAtUTC", "responseSHA256", "id", "identity"})
    require(receipt["id"] == VM and re.fullmatch(r"[a-f0-9]{64}", receipt["responseSHA256"]))
    identity = receipt["identity"]
    require(set(identity) == {"type", "principalId", "tenantId"} and identity["type"] == "SystemAssigned")
    require(all(re.fullmatch(UUID, identity[key]) for key in ("principalId", "tenantId")))
    first = timestamp(first_intent)
    observed = timestamp(receipt["observedAtUTC"])
    expiry = min(first + dt.timedelta(minutes=90), timestamp(ABSOLUTE_DEADLINE))
    now = now or dt.datetime.now(dt.timezone.utc)
    require(timestamp(APPROVED) <= first <= observed <= now < expiry)
    return {"schemaVersion": 1, "trialId": WORKFLOW, "vmId": VM,
            "principalId": identity["principalId"], "tenantId": identity["tenantId"],
            "accountId": ACCOUNT, "database": "p1", "container": "graph",
            "approvedAtUTC": APPROVED, "notAfterUTC": expiry.isoformat()}


def body(config, code):
    require(digest(code) == OBSERVER_SHA)
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    encoded = lambda value: base64.b64encode(value).decode("ascii")
    # The unchanged observer is loaded in memory. This wrapper supplies its
    # existing observe API and the same bounded/error-redacted main behavior.
    source = "\n".join([
        "set -eu", "exec /usr/bin/python3 - <<'AF_B06_OBSERVER'",
        "import base64, hashlib, json, signal, sys, types",
        "try:",
        f"    code = base64.b64decode({encoded(code)!r})",
        f"    raw = base64.b64decode({encoded(raw)!r})",
        f"    assert hashlib.sha256(code).hexdigest() == {OBSERVER_SHA!r}",
        f"    assert hashlib.sha256(raw).hexdigest() == {digest(raw)!r}",
        "    observer = types.ModuleType('b06_pinned_observer')",
        "    exec(compile(code, 'b06_pinned_observer', 'exec'), observer.__dict__)",
        "    def timed_out(_signum, _frame):",
        "        raise observer.SafeFailure('total-time-limit')",
        "    signal.signal(signal.SIGALRM, timed_out)",
        "    signal.alarm(60)",
        "    result = observer.observe(json.loads(raw))",
        f"    result.update(observerSHA256={OBSERVER_SHA!r}, configSHA256={digest(raw)!r})",
        "    print(json.dumps(result, sort_keys=True))",
        "    status = 0 if result['classification'] in ('rbac-denied', 'read-succeeded') else 2",
        "except Exception as error:",
        "    safe_reasons = {'invalid-approval-window','invalid-config-fields','invalid-identity-binding','invalid-phase','invalid-vm-binding','invalid-account-binding','invalid-source-name','approval-window-expired-or-unbounded','redirect-refused','transport-unavailable','response-limit','managed-identity-unavailable','managed-identity-binding-mismatch','total-time-limit'}",
        "    reason = str(error) if 'observer' in locals() and isinstance(error, observer.SafeFailure) and str(error) in safe_reasons else 'observer-failed'",
        "    print(json.dumps({'classification':'inconclusive','reason':reason}))",
        "    status = 2",
        "finally:",
        "    signal.alarm(0)",
        "sys.exit(status)", "AF_B06_OBSERVER", ""])
    return {"location": "japaneast", "tags": {"application": "agefreighter", "workflow": WORKFLOW,
            "purpose": "b06-cosmos-access-observation"},
            "properties": {"source": {"script": source}, "timeoutInSeconds": 60, "asyncExecution": False}}, raw


def durable(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(receipt_path, receipt_sha, first_intent, output, now=None, supplemental=False):
    require(not receipt_path.is_symlink() and receipt_path.stat().st_size <= 8192)
    raw = receipt_path.read_bytes()
    require(len(raw) <= 8192 and digest(raw) == receipt_sha)
    binding = bindings(json.loads(raw), first_intent, now)
    code = OBSERVER.read_bytes()
    require(digest(code) == OBSERVER_SHA)
    output.mkdir(mode=0o700)
    require(output.resolve() == output and output.stat().st_mode & 0o777 == 0o700)
    durable(output / "vm-identity-receipt.json", raw)
    artifacts = []
    calls = (("before", 2), ("after", 1)) if supplemental else (("before", 1), ("after", 1), ("after", 2), ("after", 3))
    for phase, number in calls:
        config = {**binding, "phase": phase + "-grant"}
        request, config_raw = body(config, code)
        name = f"af-b06-observe-{phase}-{number:02d}"
        body_raw = (json.dumps(request, indent=2) + "\n").encode()
        durable(output / (name + "-body.json"), body_raw)
        durable(output / (name + "-config.json"), config_raw)
        artifacts.append({"id": VM + "/runCommands/" + name, "phase": phase, "ordinal": number,
                          "bodyFile": name + "-body.json", "bodySHA256": digest(body_raw),
                          "configFile": name + "-config.json", "configSHA256": digest(config_raw)})
    result = {"preparedOnly": True, "executed": False, "workflow": WORKFLOW, "vmId": VM,
              "identityReceiptSHA256": receipt_sha, "observerSHA256": OBSERVER_SHA,
              "approvedAtUTC": APPROVED, "firstVMIntentUTC": first_intent,
              "notAfterUTC": binding["notAfterUTC"], "maximumBefore": 1, "maximumAfter": 1 if supplemental else 3,
              "supplementalBeforeRequiresNewApproval": supplemental,
              "maximumSecondsPerCall": 60, "automaticRetries": 0, "artifacts": artifacts,
              "dispatchCondition": "Root must check live scope/deadline and durable intent per exact ID; never replay uncertain dispatch; stop after first recognized post-grant success."}
    durable(output / "preparation.json", (json.dumps(result, indent=2) + "\n").encode())
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm-identity-receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--first-vm-intent-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-supplemental-before", action="store_true", help="NOEXEC: before-02 and unused after-01; additional before requires separate approval")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.vm_identity_receipt, args.receipt_sha256, args.first_vm_intent_utc, args.output, supplemental=args.prepare_supplemental_before)))
        return 0
    except Exception:
        print('{"preparedOnly":true,"executed":false,"error":"binding-or-preparation-refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
