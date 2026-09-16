"""Local admission tests only; not evidence of live fault/recovery success."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("observer", Path(__file__).with_name("observe-recovery-guest.py"))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class FaultAdmission(unittest.TestCase):
    def test_network_observation_binds_source_and_initial_command(self):
        actual = observer.loader_arguments("neo4j", {"source": {"type": "neo4j"}}, {"action": "migrate-source"}, "/exact/job.json", "job-id")
        self.assertEqual(actual, ["/usr/local/bin/agefreighter", "load", "/exact/job.json", "--job-id", "job-id"])

    def test_resume_observation_uses_exact_original_job(self):
        for kind in ("csv", "neo4j"):
            actual = observer.loader_arguments(kind, {"source": {"type": kind}}, {"action": "resume-migration"}, "/exact/job.json", "job-id")
            self.assertEqual(actual, ["/usr/local/bin/agefreighter", "resume", "job-id", "--job", "/exact/job.json"])

    def test_observation_rejects_wrong_source_and_action(self):
        for kind, configured, action in [("neo4j", "csv", "migrate-source"), ("neo4j", "neo4j", "migrate-csv"), ("csv", "csv", "migrate-source"), ("neo4j", "neo4j", "inventory"), ("postgresql", "postgresql", "migrate-source")]:
            with self.subTest(kind=kind, configured=configured, action=action):
                with self.assertRaises(AssertionError):
                    observer.loader_arguments(kind, {"source": {"type": configured}}, {"action": action}, "job.json", "job-id")

    def test_go_nanosecond_timestamp(self):
        self.assertEqual(observer.timestamp("2026-09-15T11:38:02.123456789Z").microsecond, 123456)

    def test_child_of_nonleader_thread(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for tid, children in ((10, ""), (11, "123"), (12, "123 456")):
                directory = root / "10" / "task" / str(tid)
                directory.mkdir(parents=True)
                (directory / "children").write_text(children)
            self.assertEqual(observer.process_children(10, root), [123, 456])

    def setUp(self):
        self.view = dict(CommittedRows=1_400_000, Status="running", RejectedRows=0,
                         SourceRejectedRows=0, checkpointAgeSeconds=1,
                         diskUsedPercent=6, swapUsedKiB=0, hostOOMKills=0,
                         memoryBytes=100_000_000, memoryEvents={"oom": 0, "oom_kill": 0})

    def test_accepts_reviewed_boundary(self):
        observer.safe_fault(self.view, 123, 1_400_000)

    def test_reboot_boundary_is_bounded(self):
        self.view["CommittedRows"] = 3_360_000
        observer.safe_fault(self.view, 123, 3_360_000)
        for rows in (3_359_999, 4_000_000):
            self.view["CommittedRows"] = rows
            with self.assertRaises(AssertionError):
                observer.safe_fault(self.view, 123, 3_360_000)

    def test_rejects_every_unsafe_boundary(self):
        changes = [("CommittedRows", 1_399_999), ("CommittedRows", 2_500_000),
                   ("Status", "committed"), ("RejectedRows", 1),
                   ("SourceRejectedRows", 1), ("checkpointAgeSeconds", 901),
                   ("checkpointAgeSeconds", -1), ("diskUsedPercent", 80),
                   ("swapUsedKiB", 1), ("hostOOMKills", 1),
                   ("memoryBytes", 4 * 1024 ** 3 + 1),
                   ("memoryEvents", {"oom": 1, "oom_kill": 0}),
                   ("memoryEvents", {"oom": 0, "oom_kill": 1})]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                view = copy.deepcopy(self.view)
                view[key] = value
                with self.assertRaises(AssertionError):
                    observer.safe_fault(view, 123, 1_400_000)

    def test_no_pid_or_unreviewed_threshold(self):
        for pid, threshold in ((None, 1_400_000), (0, 1_400_000), (123, 0)):
            with self.assertRaises(AssertionError):
                observer.safe_fault(self.view, pid, threshold)

    def test_missing_evidence_fails_closed(self):
        for key in self.view:
            view = copy.deepcopy(self.view)
            del view[key]
            with self.assertRaises(KeyError):
                observer.safe_fault(view, 123, 1_400_000)
        for key in ("oom", "oom_kill"):
            view = copy.deepcopy(self.view)
            del view["memoryEvents"][key]
            with self.assertRaises(KeyError):
                observer.safe_fault(view, 123, 1_400_000)


class NetworkFault(unittest.TestCase):
    operation = "11111111-1111-4111-8111-111111111111"

    def setUp(self):
        self.config = {"source": {"type": "neo4j", "neo4j": {"uri": "neo4j+s://source.internal:7687"}}}
        self.group = "/system.slice/agefreighter-assessment-" + self.operation + ".service"
        self.view = dict(CommittedRows=1_400_000, Status="running", RejectedRows=0,
                         SourceRejectedRows=0, checkpointAgeSeconds=1,
                         diskUsedPercent=6, swapUsedKiB=0, hostOOMKills=0,
                         memoryBytes=100_000_000, memoryEvents={"oom": 0, "oom_kill": 0},
                         ID="job", ConfigFingerprint="fingerprint", GraphGenerationID="generation",
                         bootId="boot", configSha256="hash", unit={"ControlGroup": self.group})
        self.commands = []
        self.present = False
        self.timer = False

    def fake_run(self, *args, **kwargs):
        self.commands.append(args)
        if args[0] == "systemd-run":
            self.timer = True
            return SimpleNamespace(returncode=0, stdout="")
        if args[0] == "systemctl":
            return SimpleNamespace(returncode=0, stdout="active\n")
        self.assertEqual(args[0], "/usr/sbin/iptables")
        action = args[3]
        if action == "-C":
            return SimpleNamespace(returncode=0 if self.present else 1, stdout="")
        if action == "-I":
            self.assertTrue(self.timer, "Independent restoration must be armed first")
            self.present = True
        elif action == "-D":
            self.present = False
        else:
            self.fail("Unexpected firewall action")
        return SimpleNamespace(returncode=0, stdout="")

    def invoke(self, directory, observed=None):
        with patch.object(observer, "run", side_effect=self.fake_run), \
             patch.object(observer.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("10.246.5.4", 7687))]), \
             patch.object(observer.shutil, "which", return_value="/usr/sbin/iptables"), \
             patch.object(observer.time, "sleep"):
            observer.network_fault(directory, self.config, "10.246.5.4", self.operation,
                                   self.view, 123, lambda: (observed or self.view, 123))

    def test_rule_only_matches_exact_source_and_operation(self):
        rule = observer.network_rule(self.config, "10.246.5.4", self.operation, self.group)
        self.assertEqual(rule[:7], ["OUTPUT", "-d", "10.246.5.4", "-p", "tcp", "--dport", "7687"])
        self.assertIn(self.group.lstrip("/"), rule)
        for address, group in [("1.1.1.1", self.group), ("10.246.20.4", self.group),
                               ("10.246.5.5", self.group), ("10.246.5.4", "/system.slice/other.service")]:
            with self.assertRaises(AssertionError):
                observer.network_rule(self.config, address, self.operation, group)

    def test_rejects_wrong_source_transport_port_and_credentials(self):
        for uri in ("bolt://source:7687", "neo4j+s://source:5432", "neo4j+s://user:secret@source:7687"):
            self.config["source"]["neo4j"]["uri"] = uri
            with self.assertRaises(AssertionError):
                observer.network_rule(self.config, "10.246.5.4", self.operation, self.group)

    def test_armed_before_rule_and_removed_with_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.invoke(directory)
            self.assertFalse(self.present)
            self.assertTrue((directory / "qualification-network-applied.json").is_file())
            self.assertTrue(observer.json_read(directory / "qualification-network-restored.json")["ruleAbsent"])
            count = len(self.commands)
            with self.assertRaises(FileExistsError):
                self.invoke(directory)
            self.assertFalse(any("-I" in c for c in self.commands[count:]))

    def test_changed_identity_refuses_insertion(self):
        changed = copy.deepcopy(self.view)
        changed["ConfigFingerprint"] = "other"
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(AssertionError):
                self.invoke(Path(name), changed)
            self.assertFalse(any("-I" in c for c in self.commands))

    def test_post_insert_failure_still_removes_rule(self):
        original = observer.seal
        def failing(path, value):
            if path.name == "qualification-network-applied.json":
                raise OSError("simulated evidence write failure")
            return original(path, value)
        with tempfile.TemporaryDirectory() as name, patch.object(observer, "seal", side_effect=failing):
            with self.assertRaises(OSError):
                self.invoke(Path(name))
            self.assertFalse(self.present)
            self.assertTrue(any("-D" in c for c in self.commands))

    def test_timer_failure_never_inserts(self):
        original = self.fake_run
        def fail_timer(*args, **kwargs):
            if args[0] == "systemd-run":
                return SimpleNamespace(returncode=1, stdout="")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as name, patch.object(self, "fake_run", side_effect=fail_timer):
            with self.assertRaises(AssertionError):
                self.invoke(Path(name))
            self.assertFalse(any("-I" in c for c in self.commands))

    def test_expired_restore_margin_never_inserts(self):
        with tempfile.TemporaryDirectory() as name, patch.object(observer.time, "monotonic", side_effect=[0, 16]):
            with self.assertRaises(AssertionError):
                self.invoke(Path(name))
            self.assertFalse(any("-I" in c for c in self.commands))

    def test_removal_failure_is_not_reported_restored(self):
        original = self.fake_run
        def fail_delete(*args, **kwargs):
            if args[0] == "/usr/sbin/iptables" and args[3] == "-D":
                return SimpleNamespace(returncode=2, stdout="")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as name, patch.object(self, "fake_run", side_effect=fail_delete):
            with self.assertRaises(AssertionError):
                self.invoke(Path(name))
            self.assertTrue(self.timer)
            self.assertTrue(self.present)
            self.assertFalse((Path(name) / "qualification-network-restored.json").exists())


if __name__ == "__main__":
    unittest.main()
