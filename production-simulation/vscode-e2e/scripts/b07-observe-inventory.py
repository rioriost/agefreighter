#!/usr/bin/env python3
"""NOEXEC preparation module for two exact-VM B07 process observations.

The local body builder embeds this module and its unchanged, hash-pinned B10
dependency. Importing either module performs no observation. No launcher,
worker signal, source query, token, environment or command-line read exists.
"""
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

WORKFLOW = "0f83d520-cd03-4792-8e38-3f146abfde39"
VM_ID = "/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-af-vscode-p1-20260905-a/providers/Microsoft.Compute/virtualMachines/af-0f83d520cd0347928e38"
BASE_SHA256 = "0d35975ef67274123a481a57a37fccf6f5df8683ad02bf1ea62fa7ca5b170e37"
UUID = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
SHA = re.compile(r"^[a-f0-9]{64}$")
IMDS = "http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text"


def require(value):
    if not value:
        raise ValueError("Observation binding unavailable")


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() == dt.timedelta(0))
    return parsed.timestamp()


def validate(binding):
    require(isinstance(binding, dict) and set(binding) == {
        "schemaVersion", "workflow", "vmId", "operation", "bootId",
        "guestConfigurationSHA256", "approvedAtUTC", "firstVMIntentUTC", "notAfterUTC"})
    require(type(binding["schemaVersion"]) is int and binding["schemaVersion"] == 1)
    require(binding["workflow"] == WORKFLOW and binding["vmId"] == VM_ID)
    for key in ("operation", "bootId"):
        require(isinstance(binding[key], str) and UUID.fullmatch(binding[key]))
    require(isinstance(binding["guestConfigurationSHA256"], str) and SHA.fullmatch(binding["guestConfigurationSHA256"]))
    approved = timestamp(binding["approvedAtUTC"])
    first_intent = timestamp(binding["firstVMIntentUTC"])
    expires = timestamp(binding["notAfterUTC"])
    require(approved <= first_intent < expires)
    require(expires - first_intent <= 90 * 60 and expires - approved <= 4 * 60 * 60)
    return binding


def binding_sha(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError("Redirect refused")


def guest_backend(base):
    class Backend(base.ReadOnlyBackend):
        def __init__(self):
            self.started = time.monotonic()

        def check(self):
            require(time.monotonic() - self.started < 45)

        def read(self, path, limit=8192):
            self.check()
            result = super().read(path, limit)
            self.check()
            return result

        def run(self, args):
            self.check()
            require(args[:2] == ["/bin/systemctl", "show"])
            result = super().run(args)  # Existing bounded five-second diagnostic.
            self.check()
            return result

        def resource_id(self):
            self.check()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            request = urllib.request.Request(IMDS, headers={"Metadata": "true"}, method="GET")
            with opener.open(request, timeout=5) as response:
                require(response.status == 200 and response.geturl() == IMDS)
                data = response.read(2049)
            self.check()
            require(len(data) <= 2048)
            return data.decode("utf-8").strip()

    return Backend()


def state(binding, backend):
    # This file contains durable worker identity/state, not its job or secrets.
    path = f"/var/lib/agefreighter/workflows/{WORKFLOW}/{binding['operation']}/state.json"
    value = json.loads(backend.read(path, 32768))
    require(isinstance(value, dict))
    expected = {"version": 1, "workflow": WORKFLOW, "operation": binding["operation"],
                "action": "inventory", "phase": "running", "bootId": binding["bootId"],
                "configSha256": binding["guestConfigurationSHA256"]}
    require(all(type(value.get(k)) is type(v) and value.get(k) == v for k, v in expected.items()))
    return expected  # Never copy any unrecognized state fields or error text.


def process_identity(value):
    return {k: value[k] for k in ("pid", "ppid", "startTicks", "rssBytes", "executable", "state")}


def observe(binding, phase, backend, base):
    validate(binding)
    require(phase in ("before", "after"))
    start = backend.utc()
    require(timestamp(binding["firstVMIntentUTC"]) <= timestamp(start) < timestamp(binding["notAfterUTC"]))
    require(backend.resource_id().lower() == VM_ID.lower())
    boot = backend.read("/proc/sys/kernel/random/boot_id", 128).strip()
    require(boot == binding["bootId"])
    first_state = state(binding, backend)
    unit = "agefreighter-assessment-" + binding["operation"] + ".service"
    group = "/system.slice/" + unit
    first = base.service(backend, unit)
    main = base.process(backend, first["MainPID"], base.TOOLS, group)
    require(first["exactControlGroup"] and main["expectedExecutable"] and main["exactUnitCgroupMember"])
    scan = base.inventory_children(backend, main, group)
    require(len(scan["children"]) == 1)
    child = scan["children"][0]
    last = base.service(backend, unit)
    main_after = base.process(backend, main["pid"], base.TOOLS, group)
    child_after = base.process(backend, child["pid"], base.CLI, group)
    last_state = state(binding, backend)
    boot_after = backend.read("/proc/sys/kernel/random/boot_id", 128).strip()
    require(boot_after == boot and first_state == last_state)
    require(first["invocationId"] and first["invocationId"] == last["invocationId"])
    require(first["MainPID"] == last["MainPID"] == main["pid"])
    for key, expected in (("loadState", "loaded"), ("activeState", "active"), ("subState", "running"),
                          ("restartDisabled", True), ("NRestarts", 0), ("exactControlGroup", True)):
        require(first[key] == last[key] == expected)
    for value in (main, main_after, child, child_after):
        require(value["expectedExecutable"] and value["exactUnitCgroupMember"])
    require(child["ppid"] == child_after["ppid"] == main["pid"])
    require(main["startTicks"] == main_after["startTicks"] and child["startTicks"] == child_after["startTicks"])
    end = backend.utc()
    require(timestamp(start) <= timestamp(end) < timestamp(binding["notAfterUTC"]))
    require(timestamp(end) - timestamp(start) < 45)
    return {"schemaVersion": 1, "readOnly": True, "phase": phase, "workflow": WORKFLOW,
            "vmId": VM_ID, "operation": binding["operation"], "bootId": boot,
            "bindingSHA256": binding_sha(binding), "guestConfigurationSHA256": binding["guestConfigurationSHA256"],
            "startedUTC": start, "finishedUTC": end, "unit": unit,
            "invocationId": first["invocationId"], "cgroupProcessCount": scan["count"],
            "mainBefore": process_identity(main), "mainAfter": process_identity(main_after),
            "childBefore": process_identity(child), "childAfter": process_identity(child_after),
            "stateBefore": "running", "stateAfter": "running", "activeInventoryProcessProven": True,
            "limitation": "Two bounded snapshots only; no continuous monitoring, completion, source data or independent health claim."}


def run_once(binding, phase, base):
    try:
        require(sys.platform == "linux" and os.geteuid() == 0)
        result = observe(binding, phase, guest_backend(base), base)
        output = json.dumps(result, separators=(",", ":"), allow_nan=False)
        require(len(output.encode()) + 1 <= 4096)
        print(output)
        return 0
    except Exception:
        print(json.dumps({"schemaVersion": 1, "readOnly": True, "activeInventoryProcessProven": False,
                          "error": "observation-inconclusive", "retryAuthorized": False}))
        return 2


if __name__ == "__main__":
    print('{"preparedOnly":true,"executed":false,"requiresReviewedBoundBody":true}')
    sys.exit(2)
