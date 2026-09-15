#!/usr/bin/env python3
"""Guest-only P1 observer; read-only unless an explicit fault option is given.

Run as root on the exact reviewed runner through Azure Run Command. Credentials
remain in the existing operation directory and child environment, never output.
The operator must first check cloud ownership, governance, budget and deadline.
--sigterm-at selects the first process fault; --reboot-at is only valid for a
resumed operation and requests one guest reboot after sealing current evidence.
This is qualification tooling, not a migration or automatic recovery mechanism.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

if not __debug__:
    raise SystemExit("Do not disable qualification assertions with -O")


def safe_fault(view, pid, threshold):
    """Fail closed on any changed, missing, or unhealthy fault prerequisite."""
    assert threshold in (1_400_000, 3_360_000)
    upper = 2_500_000 if threshold == 1_400_000 else 4_000_000
    assert threshold <= view["CommittedRows"] < upper
    assert view["Status"] == "running" and pid
    assert view["RejectedRows"] == view["SourceRejectedRows"] == 0
    assert 0 <= view["checkpointAgeSeconds"] <= 900
    assert view["diskUsedPercent"] < 80
    assert view["swapUsedKiB"] == view["hostOOMKills"] == 0
    assert view["memoryBytes"] <= 4 * 1024 ** 3
    assert view["memoryEvents"]["oom"] == view["memoryEvents"]["oom_kill"] == 0


def json_read(path):
    return json.loads(path.read_text())


def timestamp(value):
    # Ubuntu's Python 3.10 accepts microseconds, while Go emits nanoseconds.
    value = re.sub(r"\.(\d{6})\d+(?=Z|[+-])", r".\1", value)
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def process_children(main_pid, proc_root=Path("/proc")):
    # Go may fork the CLI from a non-leader OS thread. Linux reports children
    # per thread, so checking only task/<pid>/children can miss the loader.
    children = set()
    for path in (proc_root / str(main_pid) / "task").glob("*/children"):
        try:
            children.update(int(value) for value in path.read_text().split())
        except FileNotFoundError:
            continue
    return sorted(children)


def run(*args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=20, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("workflow", "operation", "job", "config-sha256", "boot", "deadline"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--watch-seconds", type=int, default=0)
    p.add_argument("--sigterm-at", type=int, default=0)
    p.add_argument("--reboot-at", type=int, default=0)
    a = p.parse_args()
    for value in (a.workflow, a.operation, a.job, a.boot):
        assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", value)
    assert re.fullmatch(r"[0-9a-f]{64}", a.config_sha256)
    assert os.geteuid() == 0 and 0 <= a.watch_seconds <= 600
    assert a.sigterm_at in (0, 1_400_000), "Only the reviewed P1 process fault"
    assert a.reboot_at in (0, 3_360_000), "Only the reviewed P1 reboot fault"
    assert not (a.sigterm_at and a.reboot_at), "One fault per invocation"
    deadline = timestamp(a.deadline)
    root = Path("/var/lib/agefreighter/workflows") / a.workflow
    directory = root / a.operation
    unit = "agefreighter-assessment-" + a.operation + ".service"
    config = directory / "job.json"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == a.config_sha256
    # Only existing protected guest credentials; never export or print them.
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
    env.update(json_read(directory / "secrets.json"))
    stop = time.monotonic() + a.watch_seconds
    binding = None

    def observe():
        assert dt.datetime.now(dt.timezone.utc) < deadline
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == a.boot
        state = json_read(directory / "state.json")
        assert (state["workflow"], state["operation"], state["jobId"]) == (a.workflow, a.operation, a.job)
        assert state["action"] in ("migrate-csv", "resume-migration")
        assert state["configSha256"] == a.config_sha256
        assert hashlib.sha256(config.read_bytes()).hexdigest() == a.config_sha256
        result = run("/usr/local/bin/agefreighter", "status", a.job, "--target", str(config), env=env)
        if result.returncode:
            return {"ready": False, "reason": "status unavailable", "operationPhase": state["phase"]}, None
        job = json.loads(result.stdout)
        assert job["ID"] == a.job
        if state["action"] == "resume-migration":
            expected = state["resume"]
            assert job["ConfigFingerprint"] == expected["fingerprint"]
            assert str(job["GraphGenerationID"]) == expected["generationId"]
            assert job["CommittedRows"] >= int(expected["committedRows"])
        view = {key: job[key] for key in ("ID", "Status", "ConfigFingerprint", "GraphGenerationID", "CommittedRows", "RejectedRows", "SourceRejectedRows", "UpdatedAt")}
        checkpoint = timestamp(job["UpdatedAt"])
        view["checkpointAgeSeconds"] = (dt.datetime.now(dt.timezone.utc) - checkpoint).total_seconds()
        view["observedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
        view["bootId"] = a.boot
        view["configSha256"] = a.config_sha256
        result = run("systemctl", "show", unit, "--property=MainPID,ControlGroup,ActiveState,SubState")
        assert result.returncode == 0
        props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        view["unit"] = props
        main_pid = int(props["MainPID"])
        match = []
        if main_pid:
            for child in process_children(main_pid):
                proc = Path("/proc") / str(child)
                try:
                    args = proc.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
                    expected = (["/usr/local/bin/agefreighter", "load", str(config), "--job-id", a.job] if state["action"] == "migrate-csv" else ["/usr/local/bin/agefreighter", "resume", a.job, "--job", str(config)])
                    if args == expected and proc.joinpath("exe").resolve() == Path("/usr/local/bin/agefreighter").resolve():
                        match.append(int(child))
                except FileNotFoundError:
                    continue
        view["loaderPids"] = match
        stat = os.statvfs(root)
        view["diskUsedPercent"] = 100 * (stat.f_blocks - stat.f_bfree) / stat.f_blocks
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        view["swapUsedKiB"] = int(mem["SwapTotal"].split()[0]) - int(mem["SwapFree"].split()[0])
        view["hostOOMKills"] = int(dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())["oom_kill"])
        if props["ControlGroup"]:
            cg = Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/")
            view["memoryBytes"] = int(cg.joinpath("memory.current").read_text())
            view["memoryEvents"] = dict((k, int(v)) for k, v in (line.split() for line in cg.joinpath("memory.events").read_text().splitlines()))
        return view, match[0] if len(match) == 1 else None

    while True:
        view, pid = observe()
        print(json.dumps(view), flush=True)
        if "CommittedRows" in view:
            current = (view["ConfigFingerprint"], view["GraphGenerationID"])
            if binding is None:
                binding = current
            assert current == binding
            if a.reboot_at and view["CommittedRows"] >= a.reboot_at:
                assert state_action(directory) == "resume-migration"
                safe_fault(view, pid, a.reboot_at)
                again, current_pid = observe()
                assert current_pid == pid
                assert (again["ConfigFingerprint"], again["GraphGenerationID"]) == binding
                safe_fault(again, current_pid, a.reboot_at)
                evidence = directory / "qualification-reboot.json"
                with evidence.open("x") as output:
                    os.chmod(evidence, 0o600)
                    json.dump({"before": again, "action": "loader-vm-reboot"}, output)
                    output.flush()
                    os.fsync(output.fileno())
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                print(json.dumps({"rebootRequested": True, "evidenceSHA256": hashlib.sha256(evidence.read_bytes()).hexdigest()}), flush=True)
                assert run("systemctl", "reboot").returncode == 0
                return
            if a.sigterm_at and view["CommittedRows"] >= a.sigterm_at:
                safe_fault(view, pid, a.sigterm_at)
                # pidfd prevents a recycled PID from ever receiving the signal.
                descriptor = os.pidfd_open(pid)
                try:
                    again, current_pid = observe()
                    assert current_pid == pid and again["Status"] == "running"
                    assert (again["ConfigFingerprint"], again["GraphGenerationID"]) == binding
                    safe_fault(again, current_pid, a.sigterm_at)
                    evidence = directory / "qualification-sigterm.json"
                    with evidence.open("x") as output:
                        os.chmod(evidence, 0o600)
                        json.dump({"before": again, "signal": "SIGTERM", "pid": pid}, output)
                        output.flush()
                        os.fsync(output.fileno())
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                    print(json.dumps({"signalSent": True, "pid": pid, "evidenceSHA256": hashlib.sha256(evidence.read_bytes()).hexdigest()}), flush=True)
                finally:
                    os.close(descriptor)
                return
        if time.monotonic() >= stop:
            return
        time.sleep(3)


def state_action(directory):
    return json_read(directory / "state.json")["action"]


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Exceptions from parsers/processes must not leak DSNs or raw output.
        import traceback
        trace = traceback.extract_tb(error.__traceback__)
        print(json.dumps({"stopped": True, "errorType": type(error).__name__, "line": trace[-1].lineno, "function": trace[-1].name}), flush=True)
        raise SystemExit(1)
