"""Local binding tests, not live recovery evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
