import contextlib
import copy
import datetime as dt
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "guard", Path(__file__).with_name("horizondb-scale-guard.py")
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

SUB = "/subscriptions/00000000-0000-0000-0000-000000000001"
OLD = SUB + "/resourceGroups/rg-agefreighter-horizon-tiny-20261118"
NEW = SUB + "/resourceGroups/rg-agefreighter-horizon-scale-20261118"
CLUSTER = OLD + "/providers/Microsoft.HorizonDb/clusters/afht-20261118-pg17"
DISK = OLD + "/providers/Microsoft.Compute/disks/preserved"


def settings(expired=False):
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "subscriptionId": SUB.split("/")[-1], "date": "20261118",
        "deleteTargets": [NEW, CLUSTER], "preservedEvidenceDiskId": DISK,
        "startedAtUtc": (now - dt.timedelta(hours=2)).isoformat(),
        "deadlineUtc": (now + dt.timedelta(hours=-1 if expired else 2)).isoformat(),
        "baselineCostUSD": 100, "lagReserveUSD": 150,
        "deletionReserveUSD": 60, "maxHourlyUSD": 35,
    }


class FakeClient:
    def __init__(self):
        self.calls = []
        self.deleted = set()
        self.snapshots = {}
        self.snapshot_ok = True
        self.with_worker = False
        self.cost_available = True

    def request(self, method, path, body=None, missing_ok=False):
        self.calls.append((method, path))
        resource = path.split("?")[0]
        if resource.endswith("/Microsoft.CostManagement/query"):
            if not self.cost_available:
                raise RuntimeError("cost telemetry unavailable")
            return {"properties": {"columns": [{"name": "PreTaxCost"}, {"name": "Currency"}],
                                   "rows": [[100, "USD"]]}}, {}
        if resource in self.deleted:
            return None, {}
        if resource == DISK:
            return {"properties": {"provisioningState": "Succeeded"}}, {}
        if resource == NEW:
            return {"tags": {"campaign": "afhs-20261118"}}, {}
        if resource == NEW + "/resources":
            return {"value": ([{"id": NEW + "/providers/Microsoft.Compute/virtualMachines/loader",
                                "type": "Microsoft.Compute/virtualMachines"}]
                              if self.with_worker else [])}, {}
        if resource.endswith("/virtualMachines/loader"):
            return {"tags": {"campaign": "afhs-20261118"},
                    "properties": {"storageProfile": {"osDisk": {"managedDisk": {
                        "id": NEW + "/providers/Microsoft.Compute/disks/loader-os"
                    }}}}}, {}
        if resource.endswith("/disks/loader-os"):
            return {"properties": {"diskSizeGB": 128}}, {}
        if "/snapshots/" in resource:
            return self.snapshots.get(resource), {}
        if resource == CLUSTER:
            return {"tags": {"application": "agefreighter", "run": "afht-20261118"}}, {}
        raise AssertionError("unexpected request: " + method + " " + path)

    def operation(self, method, path, body=None):
        self.calls.append((method, path))
        resource = path.split("?")[0]
        if method == "DELETE":
            self.deleted.add(resource)
        if method == "PUT":
            props = copy.deepcopy(body["properties"])
            props["provisioningState"] = "Succeeded" if self.snapshot_ok else "Failed"
            self.snapshots[resource] = {"properties": props}


class ScopeAndBudgetTests(unittest.TestCase):
    def run_guard(self, cfg, client):
        with patch.object(guard, "Client", return_value=client), contextlib.redirect_stdout(io.StringIO()):
            guard.main(cfg)

    def test_exact_scope_rejects_preserved_or_prefix_matching_resources(self):
        for target in (OLD, DISK, SUB, NEW + "-unrelated", CLUSTER + "/pools/writer", CLUSTER + "/../other"):
            cfg = settings()
            cfg["deleteTargets"] = [target]
            with self.assertRaises(ValueError):
                guard.validate(cfg)

    def test_lease_and_cost_inputs_are_fail_closed(self):
        for edit in (
            {"maxHourlyUSD": 0}, {"lagReserveUSD": 0}, {"baselineCostUSD": 3700},
            {"maxHourlyUSD": float("nan")}, {"deletionReserveUSD": 0},
            {"deadlineUtc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)).isoformat()},
            {"preservedEvidenceDiskId": NEW + "/providers/Microsoft.Compute/disks/evidence"},
        ):
            cfg = settings()
            cfg.update(edit)
            with self.assertRaises(ValueError):
                guard.validate(cfg)

    def test_valid_live_lease_retains_all_resources(self):
        client = FakeClient()
        self.run_guard(settings(), client)
        self.assertEqual(client.deleted, set())

    def test_expired_lease_deletes_only_enrolled_resources(self):
        client = FakeClient()
        self.run_guard(settings(expired=True), client)
        self.assertEqual(client.deleted, {NEW, CLUSTER})

    def test_monitoring_failure_enforces_budget_safety(self):
        client = FakeClient()
        client.cost_available = False
        self.run_guard(settings(), client)
        self.assertEqual(client.deleted, {NEW, CLUSTER})

    def test_worker_stops_and_external_snapshot_precedes_deletion(self):
        client = FakeClient()
        client.with_worker = True
        self.run_guard(settings(expired=True), client)
        methods = [method for method, _ in client.calls]
        self.assertLess(methods.index("PUT"), methods.index("DELETE"))
        stop = next(i for i, (method, path) in enumerate(client.calls) if path.endswith("/deallocate?api-version=" + guard.COMPUTE_API))
        self.assertLess(stop, methods.index("PUT"))
        self.assertEqual(len(client.snapshots), 1)
        self.assertTrue(all(path.startswith(OLD + "/") for path in client.snapshots))

    def test_failed_snapshot_prevents_deletion(self):
        client = FakeClient()
        client.with_worker = True
        client.snapshot_ok = False
        with self.assertRaises(RuntimeError):
            self.run_guard(settings(expired=True), client)
        self.assertEqual(client.deleted, set())


if __name__ == "__main__":
    unittest.main()
