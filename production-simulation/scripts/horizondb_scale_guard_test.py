import contextlib
import copy
import datetime as dt
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

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
LEASE = "00000000-0000-0000-0000-000000000002"


def settings(expired=False):
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "subscriptionId": SUB.split("/")[-1], "date": "20261118",
        "leaseId": LEASE,
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
        self.snapshot_private = True
        self.with_worker = False
        self.worker_lease = LEASE
        self.worker_stop_ok = True
        self.worker_stopped = False
        self.cost_available = True

    def request(self, method, path, body=None, missing_ok=False, *, retry_read_timeout=False):
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
            return {"tags": {"campaign": "afhs-20261118", "qualificationLease": LEASE}}, {}
        if resource == NEW + "/resources":
            return {"value": ([{"id": NEW + "/providers/Microsoft.Compute/virtualMachines/loader",
                                "type": "Microsoft.Compute/virtualMachines"}]
                              if self.with_worker else [])}, {}
        if resource.endswith("/virtualMachines/loader"):
            return {"tags": {"campaign": "afhs-20261118", "qualificationLease": self.worker_lease},
                    "properties": {"storageProfile": {"osDisk": {"managedDisk": {
                        "id": NEW + "/providers/Microsoft.Compute/disks/loader-os"
                    }}}}}, {}
        if resource.endswith("/virtualMachines/loader/instanceView"):
            return {"statuses": [{"code": "PowerState/deallocated" if self.worker_stopped else "PowerState/running"}]}, {}
        if resource.endswith("/virtualMachines/loader/deallocate"):
            self.worker_stopped = self.worker_stop_ok
            return {}, {}
        if resource.endswith("/disks/loader-os"):
            return {"properties": {"diskSizeGB": 128}}, {}
        if "/snapshots/" in resource:
            if method == "PUT":
                props = copy.deepcopy(body["properties"])
                props["provisioningState"] = "Succeeded" if self.snapshot_ok else "Failed"
                if not self.snapshot_private:
                    props["publicNetworkAccess"] = "Enabled"
                self.snapshots[resource] = {"properties": props, "sku": body["sku"]}
                return {}, {}
            return self.snapshots.get(resource), {}
        if resource == CLUSTER:
            return {"tags": {"application": "agefreighter", "run": "afht-20261118", "qualificationLease": LEASE}}, {}
        raise AssertionError("unexpected request: " + method + " " + path)

    def operation(self, method, path, body=None):
        self.calls.append((method, path))
        resource = path.split("?")[0]
        if method == "DELETE":
            self.deleted.add(resource)


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
        self.assertFalse(any("/Microsoft.CostManagement/" in path for _, path in client.calls))

    def test_cost_management_specific_retry_header_is_respected(self):
        self.assertEqual(guard.retry_delay({
            "Retry-After": "1",
            "x-ms-ratelimit-microsoft.costmanagement-qpu-retry-after": "45",
            "x-ms-ratelimit-microsoft.costmanagement-entity-retry-after": "30",
        }, 0), 45)
        with self.assertRaises(RuntimeError):
            guard.retry_delay({"Retry-After": "120"}, 0)
        with self.assertRaises(ValueError):
            guard.retry_delay({"x-ms-ratelimit-microsoft.costmanagement-qpu-retry-after": "NaN"}, 0)
        self.assertEqual(guard.retry_delay({
            "Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT",
        }, 0), 1)
        with self.assertRaises(ValueError):
            guard.retry_delay({"Retry-After": "Wed, 18 Nov 2026 08:00:00"}, 0)

    def test_monitoring_failure_enforces_budget_safety(self):
        client = FakeClient()
        client.cost_available = False
        self.run_guard(settings(), client)
        self.assertEqual(client.deleted, {NEW, CLUSTER})

    def test_cost_read_transport_failures_enforce_budget_safety(self):
        for error in (TimeoutError("cost response timed out"),
                      guard.http.client.IncompleteRead(b"", 1)):
            with self.subTest(error=type(error).__name__):
                client = FakeClient()
                with patch.object(guard, "reported_cost", side_effect=error):
                    self.run_guard(settings(), client)
                self.assertEqual(client.deleted, {NEW, CLUSTER})

    def test_cost_query_recovers_after_throttling_and_one_read_timeout(self):
        client = guard.Client()
        client.token = "test-token"
        throttled = guard.urllib.error.HTTPError(guard.ARM, 429, "throttled", {"Retry-After": "3"}, io.BytesIO())
        timed_out = Mock()
        timed_out.read.side_effect = TimeoutError("read timed out")
        response = Mock(status=200, headers={})
        response.read.return_value = b'{"properties":{"columns":[{"name":"PreTaxCost"},{"name":"Currency"}],"rows":[[100,"USD"]]}}'
        stages = [throttled, contextlib.nullcontext(timed_out), contextlib.nullcontext(response)]
        with (patch.object(client, "authenticate"),
              patch.object(client.http, "open", side_effect=stages) as request,
              patch.object(guard.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(guard.reported_cost(client, settings()), 100)
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(3,), (2,)])

    def test_second_cost_query_timeout_is_not_retried(self):
        client = guard.Client()
        client.token = "test-token"
        with (patch.object(client, "authenticate"),
              patch.object(client.http, "open", side_effect=TimeoutError()) as request,
              patch.object(guard.time, "sleep"), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(TimeoutError):
                guard.reported_cost(client, settings())
        self.assertEqual(request.call_count, 2)

    def test_worker_stops_and_external_snapshot_precedes_deletion(self):
        client = FakeClient()
        client.with_worker = True
        self.run_guard(settings(expired=True), client)
        methods = [method for method, _ in client.calls]
        self.assertLess(methods.index("PUT"), methods.index("DELETE"))
        stop = next(i for i, (method, path) in enumerate(client.calls) if path.endswith("/deallocate?api-version=" + guard.COMPUTE_API))
        confirmed = next(i for i, (_, path) in enumerate(client.calls) if i > stop and "/loader/instanceView?" in path)
        self.assertLess(stop, confirmed)
        self.assertLess(confirmed, methods.index("PUT"))
        self.assertEqual(len(client.snapshots), 1)
        self.assertTrue(all(path.startswith(OLD + "/") for path in client.snapshots))

    def test_failed_snapshot_prevents_deletion(self):
        client = FakeClient()
        client.with_worker = True
        client.snapshot_ok = False
        with self.assertRaises(RuntimeError):
            self.run_guard(settings(expired=True), client)
        self.assertEqual(client.deleted, set())

    def test_snapshot_without_privacy_prevents_deletion(self):
        client = FakeClient()
        client.with_worker = True
        client.snapshot_private = False
        with self.assertRaisesRegex(RuntimeError, "snapshot identity or privacy differs"):
            self.run_guard(settings(expired=True), client)
        self.assertEqual(client.deleted, set())

    def test_worker_must_reach_deallocated_before_snapshot(self):
        client = FakeClient()
        client.with_worker = True
        client.worker_stop_ok = False
        with patch.object(guard.time, "monotonic", side_effect=[0, 1200]):
            with self.assertRaisesRegex(RuntimeError, "worker deallocation was not verified"):
                self.run_guard(settings(expired=True), client)
        self.assertEqual(client.deleted, set())
        self.assertFalse(client.snapshots)

    def test_worker_from_another_lease_is_not_stopped(self):
        client = FakeClient()
        client.with_worker = True
        client.worker_lease = "00000000-0000-0000-0000-000000000003"
        with self.assertRaisesRegex(RuntimeError, "worker ownership or execution lease differs"):
            self.run_guard(settings(expired=True), client)
        self.assertFalse(any(method == "POST" and "/deallocate?" in path for method, path in client.calls))
        self.assertEqual(client.deleted, set())

    def test_expired_job_cannot_delete_a_reenrolled_campaign(self):
        client = FakeClient()
        cfg = settings(expired=True)
        cfg["leaseId"] = "00000000-0000-0000-0000-000000000003"
        with self.assertRaises(RuntimeError):
            self.run_guard(cfg, client)
        self.assertEqual(client.deleted, set())

    def test_location_polling_accepts_documented_empty_completion_only(self):
        client = guard.Client()
        with patch.object(client, "request", side_effect=[
            ({}, {"Location": guard.ARM + "/operation"}),
            ({}, {"X-Qualification-HTTP-Status": "202"}),
            ({}, {"X-Qualification-HTTP-Status": "204"}),
        ]) as request, patch.object(guard.time, "sleep"):
            client.operation("POST", "/publish", {})
            self.assertEqual(request.call_count, 3)
        with patch.object(client, "request", side_effect=[
            ({}, {"Azure-AsyncOperation": guard.ARM + "/operation"}),
            ({}, {"X-Qualification-HTTP-Status": "200"}),
        ]), patch.object(guard.time, "sleep"):
            with self.assertRaises(RuntimeError):
                client.operation("POST", "/publish", {})

    def test_deletion_polls_owned_resource_until_final_404(self):
        client = guard.Client()
        with patch.object(client, "request", side_effect=[
            ({}, {"Azure-AsyncOperation": guard.ARM + "/unowned-global-operation"}),
            ({"properties": {"state": "Deleting"}}, {}),
            (None, {}),
        ]) as request, patch.object(guard.time, "sleep"):
            client.operation("DELETE", CLUSTER)
            self.assertEqual(request.call_args_list, [
                unittest.mock.call("DELETE", CLUSTER, None, missing_ok=True),
                unittest.mock.call("GET", CLUSTER, missing_ok=True),
                unittest.mock.call("GET", CLUSTER, missing_ok=True),
            ])
        with patch.object(client, "request", return_value=(
            {"properties": {"state": "Deleting"}}, {}
        )), patch.object(guard.time, "sleep"), patch.object(
            guard.time, "monotonic", side_effect=[0, 0, 2700]
        ):
            with self.assertRaisesRegex(RuntimeError, "not verified absent within 45 minutes"):
                client.operation("DELETE", CLUSTER)

    def test_slow_group_deletion_can_complete_after_twenty_minutes(self):
        client = guard.Client()
        with patch.object(client, "request", side_effect=[
            ({}, {"Location": guard.ARM + "/unowned-global-operation"}),
            ({"properties": {"provisioningState": "Deleting"}}, {}),
            (None, {}),
        ]) as request, patch.object(guard.time, "sleep"), patch.object(
            guard.time, "monotonic", side_effect=[0, 0, 1201]
        ):
            client.operation("DELETE", NEW)
            self.assertEqual(request.call_args_list, [
                unittest.mock.call("DELETE", NEW, None, missing_ok=True),
                unittest.mock.call("GET", NEW, missing_ok=True),
                unittest.mock.call("GET", NEW, missing_ok=True),
            ])

    def test_non_delete_lifecycle_retains_twenty_minute_bound(self):
        client = guard.Client()
        with patch.object(client, "request", return_value=(
            {}, {"Azure-AsyncOperation": guard.ARM + "/operation"}
        )) as request, patch.object(guard.time, "monotonic", side_effect=[0, 1200]):
            with self.assertRaisesRegex(RuntimeError, "lifecycle operation exceeded 20 minutes"):
                client.operation("POST", "/publish", {})
            request.assert_called_once_with("POST", "/publish", {}, missing_ok=False)


if __name__ == "__main__":
    unittest.main()
