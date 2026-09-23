#!/usr/bin/env python3
"""Local-only builder: create exactly two sealed B07 Run Command bodies.

No Azure launcher or guest execution. Operation/boot/guest configuration hash
must be copied from the actual second inventory after its normal GUI dispatch.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("b07_observer", HERE / "b07-observe-inventory.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def durable(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def bodies(binding, observer_sha, base_sha):
    observer.validate(binding)
    code = (HERE / "b07-observe-inventory.py").read_bytes()
    base = (HERE / "b10-observe-inventory.py").read_bytes()
    observer.require(digest(code) == observer_sha and digest(base) == base_sha == observer.BASE_SHA256)
    encoded = lambda value: base64.b64encode(value).decode("ascii")
    binding_bytes = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    result = {}
    for phase in ("before", "after"):
        # Embedded reviewed code runs in memory; there is no guest file write.
        source = "\n".join([
            "set -eu", "exec /usr/bin/python3 - <<'AF_B07_OBSERVER'",
            "import base64, hashlib, json, sys, types",
            f"base_bytes = base64.b64decode({encoded(base)!r})",
            f"code_bytes = base64.b64decode({encoded(code)!r})",
            f"binding_bytes = base64.b64decode({encoded(binding_bytes)!r})",
            f"assert hashlib.sha256(base_bytes).hexdigest() == {base_sha!r}",
            f"assert hashlib.sha256(code_bytes).hexdigest() == {observer_sha!r}",
            f"assert hashlib.sha256(binding_bytes).hexdigest() == {digest(binding_bytes)!r}",
            "base = types.ModuleType('b07_pinned_base')",
            "exec(compile(base_bytes, 'b07_pinned_base', 'exec'), base.__dict__)",
            "observer = types.ModuleType('b07_pinned_observer')",
            "exec(compile(code_bytes, 'b07_pinned_observer', 'exec'), observer.__dict__)",
            f"sys.exit(observer.run_once(json.loads(binding_bytes), {phase!r}, base))",
            "AF_B07_OBSERVER", ""])
        body = {"location": "japaneast", "tags": {"application": "agefreighter", "workflow": observer.WORKFLOW,
                "purpose": "b07-process-observation"},
                "properties": {"source": {"script": source}, "timeoutInSeconds": 60, "asyncExecution": False}}
        result[phase] = {"id": observer.VM_ID + "/runCommands/af-b07-" + phase + "-" + binding["operation"], "body": body}
    return result


def prepare(binding_path, binding_sha, observer_sha, base_sha, output):
    observer.require(not binding_path.is_symlink() and binding_path.stat().st_size <= 32768)
    data = binding_path.read_bytes()
    observer.require(digest(data) == binding_sha)
    binding = observer.validate(json.loads(data))
    pair = bodies(binding, observer_sha, base_sha)
    # Existing preparation is never overwritten, regardless of prior execution.
    output.mkdir(mode=0o700)
    observer.require(output.resolve() == output and output.stat().st_mode & 0o777 == 0o700)
    durable(output / "binding.json", data)
    artifacts = {}
    for phase, item in pair.items():
        raw = (json.dumps(item["body"], indent=2) + "\n").encode()
        name = phase + "-body.json"
        durable(output / name, raw)
        artifacts[phase] = {"id": item["id"], "body": name, "bodySHA256": digest(raw)}
    metadata = {"preparedOnly": True, "executed": False, "maximumCalls": 2, "maximumSecondsPerCall": 60,
                "workflow": observer.WORKFLOW, "vmId": observer.VM_ID, "operation": binding["operation"],
                "bindingFileSHA256": binding_sha, "bindingIdentitySHA256": observer.binding_sha(binding),
                "observerSHA256": observer_sha, "baseSHA256": base_sha, "artifacts": artifacts}
    durable(output / "preparation.json", (json.dumps(metadata, indent=2) + "\n").encode())
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.binding, args.binding_sha256, args.observer_sha256, args.base_sha256, args.output)))
        return 0
    except Exception:
        print('{"preparedOnly":true,"executed":false,"error":"binding-or-preparation-refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
