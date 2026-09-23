"""Offline contracts only. No Azure, IMDS, credentials or real HTTP requests."""
import base64
import datetime as dt
import importlib.util
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("b06", Path(__file__).with_name("observe-b06-cosmos-access.py"))
b06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b06)
ID = "11111111-1111-4111-8111-111111111111"
PRINCIPAL = "22222222-2222-4222-8222-222222222222"
BASE = f"/subscriptions/{ID}/resourceGroups/b06-trial/providers/"


def config():
    now = dt.datetime.now(dt.timezone.utc)
    return {"schemaVersion": 1, "trialId": ID, "phase": "before-grant", "vmId": BASE + "Microsoft.Compute/virtualMachines/dedicated",
            "principalId": PRINCIPAL, "tenantId": ID, "accountId": BASE + "Microsoft.DocumentDB/databaseAccounts/fixture",
            "database": "p1", "container": "graph", "approvedAtUTC": (now - dt.timedelta(seconds=1)).isoformat(),
            "notAfterUTC": (now + dt.timedelta(minutes=10)).isoformat()}


def fake_token(c, changes=None):
    claims = {"oid": c["principalId"], "tid": c["tenantId"], "xms_mirid": c["vmId"], "aud": "https://cosmos.azure.com", "exp": int(time.time()) + 600}
    claims.update(changes or {})
    return "fake." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".not-a-signature"


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.no_network = patch.object(b06.urllib.request.OpenerDirector, "open", side_effect=AssertionError("offline test attempted network"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def run_probe(self, status, body, changes=None):
        c = config()
        requests = []
        token = fake_token(c, changes)
        def request(method, url, headers, data=None):
            requests.append((method, url, headers, data))
            if len(requests) == 1:
                return 200, {}, json.dumps({"access_token": token}).encode()
            return status, {"x-ms-activity-id": ID, "x-ms-request-charge": "2.81", "x-ms-substatus": "5301"}, json.dumps(body).encode()
        result = b06.observe(c, request)
        return result, requests, token

    def test_success_is_one_constant_query_no_pagination_or_secret_output(self):
        result, requests, token = self.run_probe(200, {"Documents": [1], "_count": 1})
        self.assertEqual(result["classification"], "read-succeeded")
        self.assertFalse(result["guiAssessment"])
        self.assertEqual([r[0] for r in requests], ["GET", "POST"])
        self.assertEqual(requests[0][1], b06.IMDS)
        self.assertEqual(json.loads(requests[1][3]), b06.QUERY)
        self.assertEqual(requests[1][2]["x-ms-documentdb-isquery"], "True")
        self.assertEqual(requests[1][2]["x-ms-max-item-count"], "1")
        self.assertNotIn(token, json.dumps(result))
        self.assertNotIn("Authorization", json.dumps(result))
        self.assertEqual(result["automaticRetries"], 0)

    def test_explicit_same_principal_rbac_denial_is_identified_without_raw_message(self):
        message = f"Request blocked by Auth {PRINCIPAL} does not have required RBAC permissions to perform action [Microsoft.DocumentDB/databaseAccounts/readMetadata] on resource [/]."
        result, requests, _ = self.run_probe(403, {"code": "Forbidden", "message": message})
        self.assertEqual(result["classification"], "rbac-denied")
        self.assertTrue(result["servicePrincipalMatches"])
        self.assertNotIn(message, json.dumps(result))
        self.assertEqual(len(requests), 2)

    def test_arbitrary_403_and_wrong_principal_are_inconclusive(self):
        for message in ("IP blocked", "Forbidden", f"{ID} does not have required RBAC permissions to perform action Microsoft.DocumentDB/databaseAccounts/readMetadata"):
            with self.subTest(message=message):
                result, requests, _ = self.run_probe(403, {"message": message})
                self.assertEqual(result["classification"], "inconclusive")
                self.assertEqual(len(requests), 2)

    def test_no_retry_for_authentication_throttle_or_service_failure(self):
        for status in (401, 429, 500, 503):
            with self.subTest(status=status):
                result, requests, _ = self.run_probe(status, {"message": "not evidence"})
                self.assertEqual(result["classification"], "inconclusive")
                self.assertEqual(len(requests), 2)

    def test_unexpected_success_body_is_not_read_proof(self):
        for body in ({"Documents": []}, {"Documents": [True]}, {"Documents": ["sensitive-source-value"]}, {"Documents": [1, 1]}, []):
            result, _, _ = self.run_probe(200, body)
            self.assertEqual(result["classification"], "inconclusive")
            self.assertNotIn("sensitive-source-value", json.dumps(result))

    def test_binding_mismatch_never_reaches_cosmos(self):
        for change in ({"oid": ID}, {"tid": PRINCIPAL}, {"xms_mirid": BASE + "Microsoft.Compute/virtualMachines/wrong"}, {"aud": "https://management.azure.com"}, {"exp": 0}):
            c, calls = config(), []
            def request(*args):
                calls.append(args)
                return 200, {}, json.dumps({"access_token": fake_token(c, change)}).encode()
            with self.subTest(change=change), self.assertRaisesRegex(b06.SafeFailure, "binding-mismatch"):
                b06.observe(c, request)
            self.assertEqual(len(calls), 1)

    def test_invalid_or_expired_plan_cannot_contact_imds(self):
        for key, value in (("notAfterUTC", "2000-01-01T00:00:00Z"), ("phase", "retry-loop"), ("accountId", "https://evil.invalid"),
                           ("container", "../credentials"), ("principalId", "wrong"), ("schemaVersion", True), ("keys", "forbidden")):
            c = config()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(b06.SafeFailure):
                b06.observe(c, lambda *args: self.fail("invalid config attempted network"))

    def test_headers_and_error_values_never_leak_arbitrary_text(self):
        value = b06.summarize(403, {"x-ms-activity-id": "SENSITIVE", "x-ms-substatus": "SECRET", "x-ms-request-charge": "TOKEN"}, b'{"message":"PRIVATE"}', PRINCIPAL)
        for text in ("SENSITIVE", "SECRET", "TOKEN", "PRIVATE"):
            self.assertNotIn(text, json.dumps(value))

    def test_redirects_are_refused_before_forwarding_token(self):
        with self.assertRaisesRegex(b06.SafeFailure, "redirect-refused"):
            b06.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid")

    def test_phase_change_preserves_exact_identity_binding(self):
        c = config()
        def request(method, *_):
            return (200, {}, json.dumps({"access_token": fake_token(c)}).encode()) if method == "GET" else (200, {}, b'{"Documents":[1]}')
        before = b06.observe(c, request)
        c["phase"] = "after-grant"
        after = b06.observe(c, request)
        self.assertEqual(before["bindingSHA256"], after["bindingSHA256"])
        self.assertEqual(before["querySHA256"], after["querySHA256"])


if __name__ == "__main__":
    unittest.main()
