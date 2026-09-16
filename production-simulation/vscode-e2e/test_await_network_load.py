"""Local binding tests, not live recovery evidence."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("watcher", Path(__file__).with_name("await-network-load.py"))
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class Binding(unittest.TestCase):
    def setUp(self):
        self.state = dict(version=1, workflow=watcher.WORKFLOW,
                          operation="11111111-1111-4111-8111-111111111111",
                          jobId="11111111-1111-4111-8111-111111111111",
                          action="migrate-source", bootId="boot", phase="running")
        self.config = {"source": {"type": "neo4j", "neo4j": {
            "uri": "neo4j+s://neo4j526.azn526.internal:7687", "database": "neo4j",
            "sourceId": watcher.WORKFLOW}}, "target": {"graph": watcher.GRAPH}}

    def test_exact_binding(self):
        watcher.binding(self.state, self.config, "boot")

    def test_completed_r1_is_never_a_new_fault_target(self):
        previous = "8a9ae99e-c621-4a94-afd1-a30ff210a201"
        with self.assertRaises(AssertionError):
            watcher.binding({**self.state, "workflow": previous}, self.config, "boot")
        changed = copy.deepcopy(self.config)
        changed["target"]["graph"] = "neo4j526_network_recovery_p1_r1"
        with self.assertRaises(AssertionError):
            watcher.binding(self.state, changed, "boot")

    def test_refuses_changed_operation_source_or_graph(self):
        for key, value in (("workflow", "other"), ("jobId", "other"), ("phase", "finished"),
                           ("bootId", "other"), ("operation", "../../other"), ("action", "resume-migration")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                watcher.binding({**self.state, key: value}, self.config, "boot")
        for path in (("source", "type"), ("source", "neo4j", "uri"),
                     ("source", "neo4j", "database"), ("source", "neo4j", "sourceId"), ("target", "graph")):
            changed = copy.deepcopy(self.config)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "other"
            with self.subTest(path=path), self.assertRaises(AssertionError):
                watcher.binding(self.state, changed, "boot")

    def test_inventory_is_not_a_load_and_multiple_loads_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for i, action in enumerate(("inventory", "migrate-source", "resume-migration")):
                directory = root / str(i)
                directory.mkdir()
                (directory / "state.json").write_text(json.dumps({"action": action}))
                if i < 2:
                    self.assertEqual(len(watcher.candidates(root)), i)
                else:
                    with self.assertRaises(AssertionError):
                        watcher.candidates(root)

    def test_empty_failure_requires_fresh_bound_proof_and_retains_failed_operation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            previous = root / watcher.EMPTY_FAILURE
            previous.mkdir()
            state = dict(workflow=watcher.WORKFLOW, operation=watcher.EMPTY_FAILURE,
                         jobId=watcher.EMPTY_FAILURE, action="migrate-source", phase="failed",
                         exitCode=1, finishedAt="2026-09-16T09:57:38+00:00",
                         configSha256="19ce9281467d7969ae733bae303ae27471f961c2ecade0610b2e721cd42038f5")
            (previous / "state.json").write_text(json.dumps(state))
            (previous / "job.json").write_bytes(b"config")
            (previous / "load.stderr.log").write_bytes(b"auth-error")
            (previous / "load.json").write_bytes(b"")
            proof = root / "diagnostic-11111111-1111-4111-8111-111111111111" / "doctor.json"
            proof.parent.mkdir()
            doc = dict(command="doctor", generatedAt="2026-09-16T10:00:00+00:00", errors=[], checks=[
                dict(id="metadata-schema", status="unavailable", detail="installed=0 supported=21 pending=0; doctor does not migrate"),
                dict(id="target-graph", status="pass", summary='target graph "' + watcher.GRAPH + '" is absent')])
            proof.write_text(json.dumps(doc))
            real_sha = hashlib.sha256
            digest = real_sha(proof.read_bytes()).hexdigest()
            class Digest:
                def __init__(self, value): self.value = value
                def hexdigest(self): return self.value
            def synthetic_sha(data):
                if data == b"config": return Digest(state["configSha256"])
                if data == b"auth-error": return Digest("7cade80c58ef868a3d8b00a76bc73129025d8d798ff35e2dce68d472bc222875")
                return real_sha(data)
            now = dt.datetime.fromisoformat("2026-09-16T10:01:00+00:00")
            with patch.object(watcher.hashlib, "sha256", synthetic_sha):
                self.assertEqual(watcher.empty_failure(root, proof, digest, now, dt.datetime.fromisoformat), watcher.EMPTY_FAILURE)
                for bad_digest, bad_now in [("0" * 64, now), (digest, now+dt.timedelta(minutes=16)), (digest, now-dt.timedelta(minutes=2))]:
                    with self.assertRaises(AssertionError): watcher.empty_failure(root, proof, bad_digest, bad_now, dt.datetime.fromisoformat)
                (root / "active").write_text("another")
                with self.assertRaises(AssertionError): watcher.empty_failure(root, proof, digest, now, dt.datetime.fromisoformat)
            self.assertEqual(len(watcher.candidates(root)), 1)
            self.assertEqual(watcher.candidates(root, watcher.EMPTY_FAILURE), [])
            self.assertTrue((previous / "state.json").exists())
            state["phase"] = "running"
            (previous / "state.json").write_text(json.dumps(state))
            with self.assertRaises(AssertionError): watcher.candidates(root, watcher.EMPTY_FAILURE)


if __name__ == "__main__":
    unittest.main()
