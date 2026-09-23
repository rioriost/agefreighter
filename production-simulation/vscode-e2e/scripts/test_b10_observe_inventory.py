"""Offline structural/mocked tests only: never inspect this host or a guest."""
import ast
import importlib.util
import json
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).with_name("b10-observe-inventory.py")
SPEC = importlib.util.spec_from_file_location("b10_observer", SCRIPT)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)
OP = "11111111-1111-4111-8111-111111111111"
BOOT = "22222222-2222-4222-8222-222222222222"
UNIT = "agefreighter-assessment-" + OP + ".service"
GROUP = "/system.slice/" + UNIT


def stat(pid, parent, start="100", rss="256"):
    values = ["S", str(parent)] + ["0"] * 17 + [start, "4096", rss]
    return str(pid) + " (PRIVATE comm (not emitted)) " + " ".join(values)


class FakeBackend:
    def __init__(self):
        self.reads, self.commands = [], []
        self.files = {
            "/proc/sys/kernel/random/boot_id": BOOT,
            "/proc/101/stat": stat(101, 1), "/proc/102/stat": stat(102, 101, "200"),
            "/proc/101/cgroup": "0::" + GROUP, "/proc/102/cgroup": "0::" + GROUP,
            "/proc/101/task/101/children": "102",
            "/proc/meminfo": "SwapTotal: 0 kB\nSwapFree: 0 kB\nPRIVATE: discarded\n",
            "/sys/fs/cgroup" + GROUP + "/memory.current": "1048576",
            "/sys/fs/cgroup" + GROUP + "/memory.peak": "2097152",
            "/sys/fs/cgroup" + GROUP + "/memory.swap.current": "0",
            "/sys/fs/cgroup" + GROUP + "/memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
        }
        self.links = {"/proc/101/exe": observer.TOOLS, "/proc/102/exe": observer.CLI}
        self.properties = {"Id": UNIT, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                           "MainPID": "101", "ControlPID": "0", "InvocationID": "a" * 32, "NRestarts": "0", "Restart": "no",
                           "ControlGroup": GROUP, "MemoryCurrent": "1048576", "MemorySwapCurrent": "0",
                           "MemoryMax": str(4 * 1024**3), "MemorySwapMax": "0", "Result": "success",
                           "ExecStart": "PRIVATE must never escape"}
        self.journal = (1, "", "")

    def read(self, path, _limit=8192):
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError("PRIVATE path details")
        value = self.files[path]
        return value() if callable(value) else value

    def link(self, path):
        return self.links.get(path, "/PRIVATE/unexpected/executable")

    def pagesize(self):
        return 4096

    def disk_percent(self):
        return 4

    def utc(self):
        return "2026-09-23T12:45:00+00:00"

    def run(self, args):
        self.commands.append(args)
        if args[0] == "/usr/bin/journalctl":
            return self.journal
        assert args[0:3] == ["/bin/systemctl", "show", UNIT]
        return 0, "\n".join(key + "=" + value for key, value in self.properties.items()), ""


class ObserverTests(unittest.TestCase):
    def test_live_same_unit_and_direct_child_are_proven_without_sensitive_output(self):
        backend = FakeBackend()
        result = observer.observe(OP, backend)
        self.assertTrue(result["activeInventoryProcessProven"])
        self.assertTrue(result["healthWithinObservedBounds"])
        self.assertEqual(result["mainBefore"]["startTicks"], "100")
        self.assertEqual(result["inventoryChildAfter"]["startTicks"], "200")
        self.assertEqual(result["inventoryChildAfter"]["rssBytes"], 1048576)
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertLess(len(json.dumps(result, separators=(",", ":")).encode()), 4096)
        self.assertFalse(any(any(term in path for term in ("environ", "cmdline", "job.json", "secrets.json", "state.json")) for path in backend.reads))
        self.assertEqual(len(backend.commands), 3)

    def test_invalid_uuid_refused_before_any_observation(self):
        for op in ("", "../other", OP + ".service", "ABCDEFAB-1111-4111-8111-111111111111"):
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                observer.observe(op, backend)
            self.assertEqual(backend.reads, [])
            self.assertEqual(backend.commands, [])

    def test_running_unit_without_inventory_child_is_not_worker_proof(self):
        for mutation in (lambda b: b.files.update({"/proc/101/task/101/children": ""}),
                         lambda b: b.links.update({"/proc/102/exe": "/unexpected"}),
                         lambda b: b.files.update({"/proc/102/stat": stat(102, 999)}),
                         lambda b: b.files.update({"/proc/102/cgroup": "0::/different"}),
                         lambda b: b.properties.update({"MainPID": "0"}),
                         lambda b: b.properties.update({"Id": "different.service"})):
            backend = FakeBackend(); mutation(backend)
            result = observer.observe(OP, backend)
            self.assertFalse(result["activeInventoryProcessProven"])
            self.assertFalse(result["healthWithinObservedBounds"])

    def test_pid_reuse_between_snapshots_refuses_proof(self):
        backend = FakeBackend()
        values = iter([stat(102, 101, "200"), stat(102, 101, "201")])
        backend.files["/proc/102/stat"] = lambda: next(values)
        self.assertFalse(observer.observe(OP, backend)["activeInventoryProcessProven"])

    def test_unit_invocation_change_refuses_proof(self):
        backend = FakeBackend()
        run = backend.run
        def changed(args):
            if args[0] == "/bin/systemctl" and backend.commands:
                backend.properties["InvocationID"] = "b" * 32
            return run(args)
        backend.run = changed
        self.assertFalse(observer.observe(OP, backend)["activeInventoryProcessProven"])

    def test_journal_unknown_not_zero_and_raw_text_never_emitted(self):
        for journal in ((1, "", "PRIVATE denied"), (0, "unrecognized PRIVATE text", ""), (2, "", "")):
            backend = FakeBackend(); backend.journal = journal
            result = observer.observe(OP, backend)
            self.assertIsNone(result["kernelOOMMatchingLineCount"])
            self.assertFalse(result["healthWithinObservedBounds"])
            self.assertNotIn("PRIVATE", json.dumps(result))
        backend = FakeBackend(); backend.journal = (0, "Killed process PRIVATE\nOut of memory PRIVATE\n", "")
        result = observer.observe(OP, backend)
        self.assertEqual(result["kernelOOMMatchingLineCount"], 2)
        self.assertFalse(result["healthWithinObservedBounds"])
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_missing_cgroup_memory_stays_unknown(self):
        backend = FakeBackend()
        del backend.files["/sys/fs/cgroup" + GROUP + "/memory.current"]
        result = observer.observe(OP, backend)
        self.assertTrue(result["activeInventoryProcessProven"])
        self.assertIsNone(result["cgroupMemory"]["memory.current"])
        self.assertFalse(result["healthWithinObservedBounds"])

    def test_swap_oom_and_large_rss_refuse_clean_health(self):
        for mutation in (lambda b: b.files.update({"/proc/meminfo": "SwapTotal: 2 kB\nSwapFree: 1 kB\n"}),
                         lambda b: b.files.update({"/sys/fs/cgroup" + GROUP + "/memory.events": "oom 1\noom_kill 0"}),
                         lambda b: b.files.update({"/proc/102/stat": stat(102, 101, rss=str(2 * 1024**2))})):
            backend = FakeBackend(); mutation(backend)
            self.assertFalse(observer.observe(OP, backend)["healthWithinObservedBounds"])

    def test_source_structure_has_only_read_only_commands_and_no_worker_signalling(self):
        tree = ast.parse(SCRIPT.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for node in calls:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self.assertNotIn((node.func.value.id, node.func.attr), (("os", "kill"), ("os", "remove"), ("os", "unlink"), ("os", "mkdir"), ("os", "rename")))
        strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        diagnostic_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "backend" and node.func.attr == "run"]
        prefixes = [[element.value for element in node.args[0].elts[:2]] for node in diagnostic_calls]
        self.assertEqual(prefixes, [["/bin/systemctl", "show"], ["/usr/bin/journalctl", "-k"]])
        self.assertTrue(any("No assessment state/configuration or credentials were read" in text for text in strings))


if __name__ == "__main__":
    unittest.main()
