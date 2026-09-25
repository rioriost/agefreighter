"""Synthetic local checker tests; never call ARM or qualify live evidence."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("checker", Path(__file__).parent / "scripts/b09-check-arm-evidence.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ARMEvidenceTest(unittest.TestCase):
    def observer(self):
        observed = {key: True for key in checker.BOOLEAN_FIELDS}
        observed.update(schemaVersion=1, scope="B09 dedicated negative-fixture guest observation", bootId="11111111-1111-4111-8111-111111111111", cloudInitStatus="error", cloudInitExitCode=1, archiveSHA256=checker.ARCHIVE, runnerExecutableProcessCount=0)
        return {"id": sorted(checker.OBSERVERS)[0], "properties": {"instanceView": {"executionState": "Succeeded", "exitCode": 0, "output": json.dumps(observed)}}}

    def test_deployment_identity_and_operations_are_exact(self):
        raw = {"id": checker.DEPLOYMENT, "properties": {"provisioningState": "Succeeded", "privateText": "DO NOT EMIT"}}
        operations = {"value": [{"properties": {"targetResource": {"id": checker.VM}, "provisioningState": "Succeeded", "privateText": "DO NOT EMIT"}}]}
        result = checker.check_deployment(raw, operations)
        self.assertTrue(result["checkPassed"])
        self.assertNotIn("DO NOT EMIT", str(result))
        raw["id"] += "foreign"
        with self.assertRaises(ValueError):
            checker.check_deployment(raw)

    def test_incomplete_or_foreign_operations_do_not_pass(self):
        raw = {"id": checker.DEPLOYMENT, "properties": {"provisioningState": "Succeeded"}}
        for operations in ({"value": [], "nextLink": "private"}, {"value": [{"properties": {"targetResource": {"id": "/foreign"}}}]}):
            with self.assertRaises(ValueError):
                checker.check_deployment(raw, operations)
        self.assertFalse(checker.check_deployment(raw, {"value": []})["checkPassed"])

    def test_exact_terminal_observer_passes_but_claim_alone_does_not(self):
        raw = self.observer()
        self.assertTrue(checker.check_observer(raw)["checkPassed"])
        output = json.loads(raw["properties"]["instanceView"]["output"])
        output["cloudInitStatus"] = "running"
        raw["properties"]["instanceView"]["output"] = json.dumps(output)
        self.assertFalse(checker.check_observer(raw)["checkPassed"])

    def test_observer_foreign_control_extra_output_and_failure_refuse_credit(self):
        raw = self.observer()
        raw["id"] = checker.VM + "/runCommands/af-b09-observe-04"
        with self.assertRaises(ValueError):
            checker.check_observer(raw)
        raw = self.observer()
        output = json.loads(raw["properties"]["instanceView"]["output"])
        output["secret"] = "DO NOT EMIT"
        raw["properties"]["instanceView"]["output"] = json.dumps(output)
        with self.assertRaises(ValueError):
            checker.check_observer(raw)
        raw = self.observer()
        raw["properties"]["instanceView"]["executionState"] = "Failed"
        self.assertFalse(checker.check_observer(raw)["checkPassed"])

    def test_readiness_binds_private_record_and_no_source_state(self):
        command_id = checker.VM + "/runCommands/af-11111111-1111-4111-8111-111111111111"
        raw = {"id": command_id, "properties": {"instanceView": {"executionState": "Failed", "exitCode": 1, "error": "Linux bootstrap did not complete successfully.\nDO NOT EMIT"}}}
        record = {"id": checker.WORKFLOW, "vmId": checker.VM, "deploymentId": checker.DEPLOYMENT, "phase": "provisioned", "guestCommand": {"id": command_id, "action": "ready", "phase": "failed"}}
        result = checker.check_readiness(raw, record)
        self.assertTrue(result["checkPassed"])
        self.assertNotIn("DO NOT EMIT", str(result))
        for key in ("assessment", "migration", "guestReady"):
            changed = dict(record)
            changed[key] = {}
            self.assertFalse(checker.check_readiness(raw, changed)["checkPassed"])
        raw["id"] += "foreign"
        with self.assertRaises(ValueError):
            checker.check_readiness(raw, record)

    def test_arm_casing_normalizes_only_the_same_approved_ids(self):
        raw = {"id": checker.DEPLOYMENT.upper(), "properties": {"provisioningState": "Succeeded"}}
        operations = {"value": [{"properties": {"targetResource": {"id": checker.VM.upper()}, "provisioningState": "Succeeded"}}]}
        deployment = checker.check_deployment(raw, operations)
        self.assertTrue(deployment["checkPassed"])
        self.assertEqual(deployment["deploymentId"], checker.DEPLOYMENT)
        self.assertEqual(deployment["operationResources"][0]["resourceId"], checker.VM)
        with self.assertRaises(ValueError):
            checker.check_deployment({"id": checker.DEPLOYMENT.replace("Microsoft", "Microſoft"), "properties": {"provisioningState": "Succeeded"}})
        observation = self.observer()
        observation["id"] = observation["id"].upper()
        result = checker.check_observer(observation)
        self.assertTrue(result["checkPassed"])
        self.assertEqual(result["commandId"], sorted(checker.OBSERVERS)[0])
        command_id = checker.VM + "/runCommands/af-11111111-1111-4111-8111-111111111111"
        command = {"id": command_id.upper(), "properties": {"instanceView": {"executionState": "Failed", "exitCode": 1, "error": "Linux bootstrap did not complete successfully."}}}
        record = {"id": checker.WORKFLOW, "vmId": checker.VM.upper(), "deploymentId": checker.DEPLOYMENT.upper(), "phase": "provisioned", "guestCommand": {"id": command_id.upper(), "action": "ready", "phase": "failed"}}
        readiness = checker.check_readiness(command, record)
        self.assertTrue(readiness["checkPassed"])
        self.assertEqual(readiness["commandId"], command_id)
        command["id"] = command_id.replace("af-11111111", "af-21111111")
        with self.assertRaises(ValueError): checker.check_readiness(command, record)


if __name__ == "__main__":
    unittest.main()
