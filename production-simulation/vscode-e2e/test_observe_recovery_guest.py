"""Local admission tests only; not evidence of live fault/recovery success."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
