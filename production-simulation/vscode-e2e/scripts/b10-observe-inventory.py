#!/usr/bin/env python3
"""Read-only B10 observation for one approved workflow and exact operation.

Run only on the approved Linux runner, supplying its NEW inventory UUID.
Emits bounded allowlisted JSON, never argv/environment/configuration/secrets or
log text. This does not dispatch, restart, stop, resume or mutate a worker.
Diagnostic subprocess timeouts may terminate only their own systemctl/journalctl
child. A missing process is not interpreted as a successful finished inventory.
"""
import datetime
import json
import os
import re
import selectors
import subprocess
import sys
import time

WORKFLOW = "ae952310-5eba-42b6-9fe3-9db00e93cdac"
UUID = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
TOOLS = "/usr/local/bin/agefreighter-tools"
CLI = "/usr/local/bin/agefreighter"
PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlPID", "InvocationID", "NRestarts", "Restart",
              "ExecMainStartTimestampMonotonic", "ActiveEnterTimestampMonotonic", "MemoryCurrent", "MemoryPeak",
              "MemorySwapCurrent", "MemoryMax", "MemorySwapMax", "Result", "ControlGroup")


class Unavailable(Exception):
    pass


def integer(value):
    return int(value) if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value) else None


class ReadOnlyBackend:
    def read(self, path, limit=8192):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "rb", closefd=False) as handle:
                data = handle.read(limit + 1)
            if len(data) > limit:
                raise Unavailable()
            return data.decode("utf-8", errors="strict")
        finally:
            os.close(fd)

    def link(self, path):
        return os.readlink(path)

    def pagesize(self):
        return os.sysconf("SC_PAGE_SIZE")

    def disk_percent(self):
        disk = os.statvfs("/var/lib/agefreighter/workflows")
        if not disk.f_blocks:
            raise Unavailable()
        return 100 * (1 - disk.f_bavail / disk.f_blocks)

    def utc(self):
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def run(self, args):
        # Fixed diagnostic commands only. Bound combined stdout/stderr and time;
        # no raw subprocess output is ever copied to the result JSON.
        if args[0] not in ("/bin/systemctl", "/usr/bin/journalctl"):
            raise Unavailable()
        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        buffers = {"out": bytearray(), "err": bytearray()}
        try:
            deadline = time.monotonic() + 5
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ, "out")
                selector.register(child.stderr, selectors.EVENT_READ, "err")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise Unavailable()
                    for key, _event in selector.select(remaining):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        buffers[key.data].extend(chunk)
                        if sum(map(len, buffers.values())) > 65536:
                            raise Unavailable()
            code = child.wait(timeout=max(0.01, deadline - time.monotonic()))
            return code, bytes(buffers["out"]).decode("utf-8"), bytes(buffers["err"]).decode("utf-8")
        finally:
            if child.poll() is None:
                child.kill()  # Only this freshly spawned diagnostic, never a worker PID.
                child.wait()
            child.stdout.close()
            child.stderr.close()


def safe(call):
    try:
        return call()
    except (OSError, ValueError, UnicodeError, Unavailable, subprocess.SubprocessError):
        return None


def service(backend, unit):
    result = backend.run(["/bin/systemctl", "show", unit, "--no-pager", "--property=" + ",".join(PROPERTIES)])
    if result[0] != 0 or result[2].strip():
        raise Unavailable()
    raw = dict(line.split("=", 1) for line in result[1].splitlines() if "=" in line)
    if raw.get("Id") != unit:
        raise Unavailable()
    group = "/system.slice/" + unit
    # Return only defined enums/numbers and the independently constructed name.
    result = {"unit": unit, "loadState": raw.get("LoadState") if raw.get("LoadState") in ("loaded", "not-found", "error", "masked") else None,
              "activeState": raw.get("ActiveState") if raw.get("ActiveState") in ("active", "inactive", "failed", "activating", "deactivating") else None,
              "subState": raw.get("SubState") if raw.get("SubState") in ("running", "dead", "failed", "start", "exited", "stop") else None,
              "invocationId": raw.get("InvocationID") if re.fullmatch(r"[0-9a-f]{32}", raw.get("InvocationID", "")) else None,
              "restartDisabled": raw.get("Restart") == "no",
              "exactControlGroup": raw.get("ControlGroup") == group}
    for key in ("MainPID", "ControlPID", "NRestarts", "ExecMainStartTimestampMonotonic", "ActiveEnterTimestampMonotonic", "MemoryCurrent", "MemoryPeak", "MemorySwapCurrent", "MemoryMax", "MemorySwapMax"):
        result[key] = integer(raw.get(key))
    result["result"] = raw.get("Result") if raw.get("Result") in ("success", "exit-code", "signal", "core-dump", "timeout", "oom-kill", "resources", "start-limit-hit") else None
    return result


def process(backend, pid, expected, group):
    if not isinstance(pid, int) or pid <= 0:
        raise Unavailable()
    stat = backend.read(f"/proc/{pid}/stat", 8192)
    # comm can contain spaces/parentheses; never emit it or read cmdline/environ.
    end = stat.rfind(")")
    if end < 0 or stat.split(" ", 1)[0] != str(pid):
        raise Unavailable()
    fields = stat[end + 2:].split()
    if len(fields) < 22 or fields[0] not in ("R", "S", "D", "I"):
        raise Unavailable()
    parent, start, rss = integer(fields[1]), integer(fields[19]), integer(fields[21])
    if parent is None or start is None or rss is None:
        raise Unavailable()
    executable_matches = backend.link(f"/proc/{pid}/exe") == expected
    cgroups = backend.read(f"/proc/{pid}/cgroup", 8192).splitlines()
    member = any(line.split(":", 2)[-1] == group for line in cgroups)
    return {"pid": pid, "ppid": parent, "startTicks": str(start), "rssBytes": rss * backend.pagesize(),
            "executable": expected if executable_matches else None, "expectedExecutable": executable_matches,
            "exactUnitCgroupMember": member, "state": fields[0]}


def oom_count(backend):
    code, out, err = backend.run(["/usr/bin/journalctl", "-k", "-b", "--no-pager", "--grep=Out of memory|Killed process", "-o", "cat"])
    if code == 1 and not out and not err:
        return 0
    if code != 0 or err.strip():
        raise Unavailable()
    lines = out.splitlines()
    if any("Out of memory" not in line and "Killed process" not in line for line in lines):
        raise Unavailable()
    return len(lines)  # Matching kernel lines, not a deduplicated incident count.


def swap_bytes(backend):
    values = {}
    for line in backend.read("/proc/meminfo", 16384).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] in ("SwapTotal:", "SwapFree:") and fields[2] == "kB":
            values[fields[0]] = integer(fields[1])
    total, free = values.get("SwapTotal:"), values.get("SwapFree:")
    if total is None or free is None or free > total:
        raise Unavailable()
    return (total - free) * 1024


def cgroup_health(backend, group):
    root = "/sys/fs/cgroup" + group
    values = {name: safe(lambda name=name: integer(backend.read(root + "/" + name, 4096).strip()))
              for name in ("memory.current", "memory.peak", "memory.swap.current")}
    events = safe(lambda: backend.read(root + "/memory.events", 4096))
    parsed = {}
    if events is not None:
        for line in events.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] in ("oom", "oom_kill", "oom_group_kill"):
                parsed[fields[0]] = integer(fields[1])
    values["oom"] = parsed.get("oom")
    values["oom_kill"] = parsed.get("oom_kill")
    values["oom_group_kill"] = parsed.get("oom_group_kill")
    return values  # Missing cgroup-v2 files remain null, never inferred zero.


def observe(operation, backend):
    if not UUID.fullmatch(operation):
        raise ValueError("Exact lowercase operation UUID required")
    unit = "agefreighter-assessment-" + operation + ".service"
    group = "/system.slice/" + unit
    started = backend.utc()
    boot = safe(lambda: backend.read("/proc/sys/kernel/random/boot_id", 128).strip())
    if not isinstance(boot, str) or not UUID.fullmatch(boot):
        boot = None
    first = safe(lambda: service(backend, unit))
    main = safe(lambda: process(backend, first["MainPID"], TOOLS, group)) if first else None
    children = []
    if main:
        raw = safe(lambda: backend.read(f"/proc/{main['pid']}/task/{main['pid']}/children", 4096))
        pids = raw.split() if raw is not None else []
        if len(pids) <= 16 and all(integer(pid) is not None for pid in pids):
            for pid in pids:
                candidate = safe(lambda pid=pid: process(backend, int(pid), CLI, group))
                if candidate and candidate["expectedExecutable"] and candidate["ppid"] == main["pid"]:
                    children.append(candidate)
    disk = safe(backend.disk_percent)
    swap = safe(lambda: swap_bytes(backend))
    oom = safe(lambda: oom_count(backend))
    memory = cgroup_health(backend, group) if first and first["exactControlGroup"] else None
    last = safe(lambda: service(backend, unit))
    main_after = safe(lambda: process(backend, main["pid"], TOOLS, group)) if main else None
    child_after = safe(lambda: process(backend, children[0]["pid"], CLI, group)) if len(children) == 1 else None
    stable = bool(boot and first and last and main and main_after and child_after and len(children) == 1
                  and first["invocationId"] and first["invocationId"] == last["invocationId"]
                  and first["MainPID"] == last["MainPID"] == main["pid"]
                  and first["loadState"] == last["loadState"] == "loaded"
                  and first["activeState"] == last["activeState"] == "active"
                  and first["subState"] == last["subState"] == "running"
                  and first["restartDisabled"] and last["restartDisabled"]
                  and first["NRestarts"] == last["NRestarts"] == 0
                  and first["exactControlGroup"] and last["exactControlGroup"]
                  and main["expectedExecutable"] and main_after["expectedExecutable"]
                  and main["exactUnitCgroupMember"] and main_after["exactUnitCgroupMember"]
                  and child_after["exactUnitCgroupMember"] and child_after["expectedExecutable"]
                  and child_after["ppid"] == main["pid"] and children[0]["exactUnitCgroupMember"]
                  and main["startTicks"] == main_after["startTicks"]
                  and children[0]["startTicks"] == child_after["startTicks"])
    health = bool(stable and disk is not None and 0 <= disk < 80 and swap == 0 and oom == 0 and memory
                  and memory["memory.current"] is not None and memory["memory.current"] <= 4 * 1024**3
                  and memory["memory.swap.current"] == 0 and memory["oom"] == 0 and memory["oom_kill"] == 0
                  and first["MemoryMax"] == last["MemoryMax"] == 4 * 1024**3
                  and first["MemorySwapMax"] == last["MemorySwapMax"] == 0
                  and children[0]["rssBytes"] <= 4 * 1024**3 and child_after["rssBytes"] <= 4 * 1024**3)
    return {"schemaVersion": 1, "readOnly": True, "workflow": WORKFLOW, "operation": operation, "unit": unit,
            "startedUTC": started, "finishedUTC": backend.utc(), "bootId": boot,
            "serviceBefore": first, "serviceAfter": last, "mainBefore": main, "mainAfter": main_after,
            "inventoryChildrenBefore": children, "inventoryChildAfter": child_after,
            "activeInventoryProcessProven": stable, "healthWithinObservedBounds": health,
            "diskUsedPercent": disk, "swapUsedBytes": swap, "kernelOOMMatchingLineCount": oom, "cgroupMemory": memory,
            "limitation": "Bounded snapshots only, not continuous monitoring. Null is unknown. False process proof does not mean finished/success. No assessment state/configuration or credentials were read."}


def main():
    if len(sys.argv) != 2 or not UUID.fullmatch(sys.argv[1]):
        print('{"error":"exact-operation-uuid-required","readOnly":true}')
        return 2
    if sys.platform != "linux":
        print('{"error":"approved-linux-runner-required","readOnly":true}')
        return 2
    try:
        result = observe(sys.argv[1], ReadOnlyBackend())
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
        return 0 if result["activeInventoryProcessProven"] and result["healthWithinObservedBounds"] else 3
    except Exception:
        # Never forward arbitrary OS errors, diagnostic output or filesystem data.
        print('{"error":"observation-unavailable","readOnly":true}')
        return 2


if __name__ == "__main__":
    sys.exit(main())
