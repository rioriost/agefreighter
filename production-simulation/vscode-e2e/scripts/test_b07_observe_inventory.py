"""Offline only: no desktop/guest process, network, Azure or source inspection."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

obs = module("b07_observer", "b07-observe-inventory.py")
builder = module("b07_builder", "prepare-b07-observer.py")
fixtures = module("b10_fixtures", "test_b10_observe_inventory.py")
base = fixtures.observer

def binding():
    return {"schemaVersion": 1, "workflow": obs.WORKFLOW, "vmId": obs.VM_ID,
            "operation": fixtures.OP, "bootId": fixtures.BOOT, "guestConfigurationSHA256": "a" * 64,
            "approvedAtUTC": "2026-09-23T12:40:00Z", "firstVMIntentUTC": "2026-09-23T12:41:00Z",
            "notAfterUTC": "2026-09-23T13:00:00Z"}

def backend():
    value = fixtures.FakeBackend()
    value.resource_id = lambda: obs.VM_ID
    value.state_path = f"/var/lib/agefreighter/workflows/{obs.WORKFLOW}/{fixtures.OP}/state.json"
    value.state = {"version": 1, "workflow": obs.WORKFLOW, "operation": fixtures.OP, "action": "inventory",
                   "phase": "running", "bootId": fixtures.BOOT, "configSha256": "a" * 64,
                   "privateUnknownField": "PRIVATE must never escape"}
    value.files[value.state_path] = lambda: json.dumps(value.state)
    return value

class B07Tests(unittest.TestCase):
    def setUp(self):
        for target, name in ((base.subprocess, "Popen"), (obs.urllib.request.OpenerDirector, "open")):
            p = patch.object(target, name, side_effect=AssertionError("Offline test attempted real observation"))
            p.start(); self.addCleanup(p.stop)

    def test_exact_running_process_and_state_are_bounded_and_no_private_output(self):
        value = backend(); result = obs.observe(binding(), "before", value, base)
        self.assertTrue(result["activeInventoryProcessProven"])
        self.assertEqual(result["mainBefore"]["startTicks"], "100")
        self.assertEqual(result["childAfter"]["startTicks"], "200")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertLess(len(json.dumps(result).encode()) + 1, 4096)
        self.assertEqual(len(value.commands), 2)
        self.assertTrue(all(c[:2] == ["/bin/systemctl", "show"] for c in value.commands))
        self.assertFalse(any(any(s in p for s in ("job.json", "secrets.json", "environ", "cmdline", "meminfo")) for p in value.reads))

    def test_wrong_binding_rejected_before_backend_access(self):
        for key, bad in (("workflow", fixtures.OP), ("vmId", obs.VM_ID + "other"), ("operation", "../unsafe"),
                         ("bootId", "unknown"), ("guestConfigurationSHA256", "bad"), ("schemaVersion", True),
                         ("notAfterUTC", "2026-09-23T17:00:00Z"), ("extra", "PRIVATE")):
            value = backend(); b = binding(); b[key] = bad
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                obs.observe(b, "before", value, base)
            self.assertEqual(value.reads, []); self.assertEqual(value.commands, [])

    def test_wrong_imds_vm_and_expired_window_do_not_read_state(self):
        for wrong_vm in (True, False):
            value = backend()
            if wrong_vm: value.resource_id = lambda: obs.VM_ID + "other"
            else: value.utc = lambda: "2026-09-23T13:00:00Z"
            with self.assertRaises(ValueError): obs.observe(binding(), "before", value, base)
            self.assertEqual(value.reads, [])

    def test_actual_approval_and_compute_clocks_are_distinct_and_bounded(self):
        # Setup can consume more than 90 minutes; never relabel its approval.
        b = binding(); b["approvedAtUTC"] = "2026-09-23T10:00:00Z"
        self.assertEqual(obs.validate(b), b)
        for changes in ({"firstVMIntentUTC": "2026-09-23T09:59:59Z"},
                        {"notAfterUTC": "2026-09-23T14:11:01Z"},
                        {"approvedAtUTC": "2026-09-23T08:59:59Z"},
                        {"notAfterUTC": b["firstVMIntentUTC"]}):
            value = dict(b, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): obs.validate(value)

    def test_wrong_durable_identity_action_or_phase_refuses(self):
        for key, bad in (("workflow", fixtures.OP), ("operation", obs.WORKFLOW), ("action", "profile"),
                         ("phase", "finished"), ("bootId", fixtures.OP), ("configSha256", "b" * 64)):
            value = backend(); value.state[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): obs.observe(binding(), "before", value, base)
            self.assertEqual(value.commands, [])

    def test_missing_child_pid_reuse_restart_and_state_transition_refuse(self):
        mutations = [lambda b: b.files.update({"/sys/fs/cgroup" + fixtures.GROUP + "/cgroup.procs": "101"}),
                     lambda b: b.properties.update({"NRestarts": "1"}),
                     lambda b: b.links.update({"/proc/102/exe": "/private/wrong"})]
        for mutate in mutations:
            value = backend(); mutate(value)
            with self.assertRaises(ValueError): obs.observe(binding(), "before", value, base)
        value = backend(); stats = iter([fixtures.stat(102, 101, "200"), fixtures.stat(102, 101, "201")])
        value.files["/proc/102/stat"] = lambda: next(stats)
        with self.assertRaises(ValueError): obs.observe(binding(), "before", value, base)
        value = backend(); second = copy.deepcopy(value.state); second["phase"] = "finished"
        states = iter([value.state, second]); value.files[value.state_path] = lambda: json.dumps(next(states))
        with self.assertRaises(ValueError): obs.observe(binding(), "after", value, base)

    def test_before_after_same_binding_is_not_itself_gui_or_continuous_proof(self):
        before = obs.observe(binding(), "before", backend(), base)
        after = obs.observe(binding(), "after", backend(), base)
        self.assertEqual(before["bindingSHA256"], after["bindingSHA256"])
        self.assertEqual(before["invocationId"], after["invocationId"])
        self.assertEqual(before["childAfter"], after["childBefore"])
        self.assertIn("no continuous monitoring", before["limitation"])

    def test_builder_pins_two_exact_resource_bodies_without_execution(self):
        sha = hashlib.sha256((HERE / "b07-observe-inventory.py").read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = root / "binding.json"; source.write_text(json.dumps(binding()))
            output = root / "prepared"; expected = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = builder.prepare(source, expected, sha, obs.BASE_SHA256, output)
            self.assertEqual(metadata["maximumCalls"], 2); self.assertFalse(metadata["executed"])
            for phase in ("before", "after"):
                item = metadata["artifacts"][phase]; body_bytes = (output / item["body"]).read_bytes(); body = json.loads(body_bytes)
                self.assertTrue(item["id"].startswith(obs.VM_ID + "/runCommands/af-b07-" + phase + "-"))
                self.assertEqual(hashlib.sha256(body_bytes).hexdigest(), item["bodySHA256"])
                self.assertEqual(body["properties"]["timeoutInSeconds"], 60)
                self.assertFalse(body["properties"]["asyncExecution"])
                self.assertNotIn("protectedParameters", body["properties"])
                self.assertIn("exec /usr/bin/python3", body["properties"]["source"]["script"])
                payload_lines = body["properties"]["source"]["script"].splitlines()
                compile("\n".join(payload_lines[2:-1]), "noexec-body-syntax", "exec")
                self.assertEqual((output / item["body"]).stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError): builder.prepare(source, expected, sha, obs.BASE_SHA256, output)
            with self.assertRaises(ValueError): builder.bodies(binding(), "0" * 64, obs.BASE_SHA256)
            with self.assertRaises(ValueError): builder.bodies(binding(), sha, "0" * 64)

    def test_runtime_failure_never_emits_raw_details_or_false_pass(self):
        output = io.StringIO()
        with patch.object(obs.sys, "platform", "linux"), patch.object(obs.os, "geteuid", return_value=0), \
             patch.object(obs, "guest_backend", side_effect=RuntimeError("PRIVATE forbidden details")), contextlib.redirect_stdout(output):
            self.assertEqual(obs.run_once(binding(), "before", base), 2)
        self.assertNotIn("PRIVATE", output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["activeInventoryProcessProven"])


if __name__ == "__main__":
    unittest.main()
