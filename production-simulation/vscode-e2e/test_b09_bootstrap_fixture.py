"""Local helper tests use inert bytes; never an Azure or guest qualification."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("fixture", Path(__file__).with_name("prepare-b09-bootstrap-fixture.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class BootstrapFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, names=("agefreighter", "agefreighter-tools"), symlink=False):
        archive_path = self.root / "original.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in names:
                member = tarfile.TarInfo(name)
                data = b"inert local fixture, not executable"
                member.size = len(data)
                if symlink:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "elsewhere"
                    member.size = 0
                archive.addfile(member, None if symlink else io.BytesIO(data))
        data = archive_path.read_bytes()
        pin = dict(commit="a" * 40, sha256=fixture.digest(data), bytes=len(data))
        manifest = dict(pin, schemaVersion=1, platform="linux-amd64", version="2.4.0-dev." + "a" * 12, archive=archive_path.name)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, pin

    def test_deterministic_derivative_preserves_member_and_original(self):
        source, pin = self.source()
        before = (self.root / "original.tar.gz").read_bytes()
        first = fixture.prepare(source, self.root / "one", pin)
        second = fixture.prepare(source, self.root / "two", pin)
        self.assertEqual(first, second)
        self.assertEqual((self.root / "original.tar.gz").read_bytes(), before)
        self.assertEqual(first["retainedMember"]["sha256"], fixture.digest(b"inert local fixture, not executable"))
        with tarfile.open(self.root / "one" / first["derivative"]["archive"]) as archive:
            self.assertEqual(archive.getnames(), ["agefreighter"])
            self.assertTrue(archive.getmembers()[0].isfile())
        self.assertFalse(first["usableRunner"])
        self.assertEqual((self.root / "one").stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.root / "one" / "manifest.json").stat().st_mode & 0o777, 0o600)

    def test_existing_output_is_not_overwritten(self):
        source, pin = self.source()
        output = self.root / "exists"
        output.mkdir()
        (output / "retain").write_text("original")
        with self.assertRaises(FileExistsError):
            fixture.prepare(source, output, pin)
        self.assertEqual((output / "retain").read_text(), "original")

    def test_archive_digest_drift_is_refused(self):
        source, pin = self.source()
        pin = dict(pin, sha256="0" * 64)
        value = json.loads(source.read_text())
        value["sha256"] = pin["sha256"]
        source.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "reviewed bytes"):
            fixture.prepare(source, self.root / "output", pin)
        self.assertFalse((self.root / "output").exists())

    def test_extra_or_duplicate_members_are_refused(self):
        for names in (("agefreighter", "agefreighter-tools", "extra"), ("agefreighter", "agefreighter")):
            source, pin = self.source(names)
            with self.assertRaisesRegex(ValueError, "exactly"):
                fixture.prepare(source, self.root / "output", pin)
        self.assertFalse((self.root / "output").exists())

    def test_symlink_member_and_symlink_archive_are_refused(self):
        source, pin = self.source(symlink=True)
        with self.assertRaisesRegex(ValueError, "regular"):
            fixture.prepare(source, self.root / "output", pin)
        source, pin = self.source()
        original = self.root / "original.tar.gz"
        original.rename(self.root / "actual.tar.gz")
        original.symlink_to("actual.tar.gz")
        with self.assertRaisesRegex(ValueError, "regular"):
            fixture.prepare(source, self.root / "output", pin)


if __name__ == "__main__":
    unittest.main()
