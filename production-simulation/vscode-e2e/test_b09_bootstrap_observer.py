"""Synthetic filesystem/status tests only; these cannot qualify a real guest."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("observer", Path(__file__).with_name("observe-b09-bootstrap-failure.py"))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class BootstrapObserverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "var/lib/agefreighter"
        self.work = self.state / "install.test"
        self.work.mkdir(parents=True)
        (self.root / "proc/sys/kernel/random").mkdir(parents=True)
        (self.root / "proc/sys/kernel/random/boot_id").write_text("11111111-1111-4111-8111-111111111111\n")
        self.binary = b"inert test binary bytes"
        (self.work / "agefreighter").write_bytes(self.binary)
        (self.work / "agefreighter-tools").write_bytes(b"")
        self.archive = self.work / "archive.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            member = tarfile.TarInfo("agefreighter")
            member.size = len(self.binary)
            tar.addfile(member, io.BytesIO(self.binary))
        self.log = self.root / "var/log/cloud-init-output.log"
        self.log.parent.mkdir(parents=True)
        self.log.write_text("tar: agefreighter-tools: Not found in archive\ntar: Exiting with failure status due to previous errors\n")
        self.status = b"status: error\nprivate error detail, do not emit\n"
        self.code = 1
        self.final = b"ActiveState=failed\nSubState=failed\nResult=exit-code\n"
        self.expected_sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def observe(self):
        def translated(value):
            path = Path(value)
            return self.root / str(path).lstrip("/") if path.is_absolute() and not path.is_relative_to(self.root) else path
        def command(args, **_kwargs):
            return SimpleNamespace(stdout=self.status if args[0] == "cloud-init" else self.final, stderr=b"", returncode=self.code if args[0] == "cloud-init" else 0)
        with patch.object(observer, "Path", translated), patch.object(observer.subprocess, "run", command), patch.object(observer, "EXPECTED_ARCHIVE", self.expected_sha), patch.object(observer, "EXPECTED_BYTES", self.archive.stat().st_size), patch.object(observer, "EXPECTED_MEMBER", hashlib.sha256(self.binary).hexdigest()), patch.object(observer, "EXPECTED_MEMBER_BYTES", len(self.binary)):
            return observer.observe()

    def test_complete_fault_evidence_and_no_raw_output(self):
        result = self.observe()
        self.assertTrue(result["terminalPackagingFailureObserved"])
        self.assertNotIn("private error", str(result))
        self.assertEqual(result["runnerExecutableProcessCount"], 0)

    def test_pending_or_unknown_status_is_not_terminal(self):
        for status in (b"status: running\n", b"unrecognized\n"):
            self.status = status
            self.code = 0
            self.assertFalse(self.observe()["terminalPackagingFailureObserved"])

    def test_wrong_archive_hash_stops_before_parsing(self):
        self.expected_sha = "0" * 64
        with patch.object(observer.tarfile, "open", side_effect=AssertionError("must not parse unpinned bytes")) as parse:
            result = self.observe()
            parse.assert_not_called()
        self.assertFalse(result["terminalPackagingFailureObserved"])
        self.assertNotIn("archiveHasOnlyRetainedRegularMember", result)
        self.assertIn("observationError", result)

    def test_different_bootstrap_failure_is_not_this_case(self):
        self.log.write_text("An unrelated package download failure\n")
        self.assertFalse(self.observe()["terminalPackagingFailureObserved"])

    def test_installed_marker_or_worker_refuses_credit(self):
        marker = self.state / "bootstrap.complete"
        marker.touch()
        self.assertFalse(self.observe()["terminalPackagingFailureObserved"])
        marker.unlink()
        process = self.root / "proc/123"
        process.mkdir()
        (process / "exe").symlink_to("/usr/local/bin/agefreighter-tools")
        result = self.observe()
        self.assertEqual(result["runnerExecutableProcessCount"], 1)
        self.assertFalse(result["terminalPackagingFailureObserved"])


if __name__ == "__main__":
    unittest.main()
