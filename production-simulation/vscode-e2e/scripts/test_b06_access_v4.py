"""Offline fake identities only; all real HTTP, process and guest-file I/O blocked."""
import builtins
import contextlib
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
obs = module("b06_v4_test_observer", "b06-observe-access-v4.py")
builder = module("b06_v4_test_builder", "prepare-b06-observer-v4.py")
NOW = dt.datetime(2030, 1, 1, 1, tzinfo=dt.timezone.utc)
WF = "11111111-1111-4111-8111-111111111111"
PRINCIPAL = "22222222-2222-4222-8222-222222222222"
BASE = "/subscriptions/33333333-3333-4333-8333-333333333333/resourceGroups/offline-fixture/providers/"
OPAQUE = "PRIVATE-opaque-not-a-jwt"


def config():
    return {"schemaVersion": 2, "trialId": WF, "phase": "before-grant", "attempt": 1,
            "vmId": BASE + "Microsoft.Compute/virtualMachines/af-" + WF.replace("-", "")[:20],
            "principalId": PRINCIPAL, "tenantId": WF,
            "accountId": BASE + "Microsoft.DocumentDB/databaseAccounts/offline-fixture",
            "database": "p1", "container": "graph", "approvedAtUTC": "2030-01-01T00:00:00Z",
            "firstVMIntentUTC": "2030-01-01T00:30:00Z", "notAfterUTC": "2030-01-01T02:00:00Z",
            "vmIdentityObservedAtUTC": "2030-01-01T00:31:00Z", "vmIdentityReceiptSHA256": "a" * 64}


def envelope(token=OPAQUE):
    return {"access_token": token, "token_type": "Bearer", "resource": obs.RESOURCE,
            "expires_on": str(int(NOW.timestamp()) + 3600), "not_before": str(int(NOW.timestamp()) - 30),
            "refresh_token": "PRIVATE-ignored-field"}


class FakeTransport:
    def __init__(self, c=None):
        self.config = c or config()
        self.calls = []
        self.token = envelope()
        self.query_status = 200
        self.query_body = {"Documents": [1]}
        self.metadata = self.config["vmId"].encode()
        self.token_status = 200
        self.raise_at = None
        self.on_call = lambda _index: None
    def __call__(self, method, url, headers, body=None, maximum=65536):
        self.calls.append((method, url, headers, body, maximum))
        index = len(self.calls)
        self.on_call(index)
        if index == self.raise_at:
            raise TimeoutError("PRIVATE upstream diagnostic")
        if index == 1:
            return 200, {}, self.metadata
        if index == 2:
            return self.token_status, {}, json.dumps(self.token).encode()
        return self.query_status, {"x-ms-activity-id": WF, "x-ms-request-charge": "2.5", "private": "PRIVATE"}, json.dumps(self.query_body).encode()


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden"))
        self.network.start(); self.addCleanup(self.network.stop)
    def run_probe(self, transport=None, c=None, clock=None):
        c = c or config(); transport = transport or FakeTransport(c); lines = []
        with patch.object(obs.sys, "platform", "linux"), patch.object(obs.os, "geteuid", return_value=0):
            code = obs.run_once(c, "b" * 64, "c" * 64, transport, lines.append, clock or (lambda: NOW))
        output = "\n".join(json.dumps(line) for line in lines)
        self.assertNotIn("PRIVATE", output)
        self.assertLessEqual(sum(len(obs.encoded(line)) + 1 for line in lines), 4096)
        return code, lines[-1], transport, lines

    def test_opaque_token_reaches_one_constant_query_and_correct_identity_selector(self):
        code, result, transport, lines = self.run_probe()
        self.assertEqual(code, 0); self.assertEqual(result["classification"], "read-succeeded")
        self.assertEqual([x[0] for x in transport.calls], ["GET", "GET", "POST"])
        self.assertEqual(transport.calls[0][1], obs.IMDS_VM)
        self.assertIn("object_id=" + PRINCIPAL, transport.calls[1][1])
        self.assertEqual(json.loads(transport.calls[2][3]), {"query": "SELECT VALUE 1 FROM c", "parameters": []})
        self.assertEqual(transport.calls[2][2]["x-ms-max-item-count"], "1")
        self.assertEqual([x["stage"] for x in lines[:-1]], ["config", "vm-metadata", "token", "query"])
        self.assertTrue(result["attemptsArePreDispatchIntents"])
        self.assertEqual(result["queryAttempts"], 1); self.assertEqual(result["queryResponsesReceived"], 1)
        self.assertFalse(result["guiAssessment"])

    def test_token_shape_and_optional_claims_are_not_client_requirements(self):
        for token in ("opaque", "header.e30.signature", "header.payload-without-oid-tid-mirid-aud.signature"):
            fake = FakeTransport(); fake.token = envelope(token)
            code, result, _, _ = self.run_probe(fake)
            self.assertEqual((code, result["classification"]), (0, "read-succeeded"))

    def test_bad_envelopes_fail_at_token_without_query(self):
        changes = ({"access_token": ""}, {"access_token": "PRIVATE\r\nheader"}, {"expires_on": True},
                   {"expires_on": "0"}, {"token_type": "Other"}, {"resource": "https://wrong.invalid"},
                   {"not_before": "9999999999"}, {"access_token": "x" * 32769})
        for change in changes:
            with self.subTest(change=list(change)):
                fake = FakeTransport(); fake.token.update(change)
                code, result, fake, _ = self.run_probe(fake)
                self.assertEqual(code, 2); self.assertEqual(result["stage"], "token")
                self.assertEqual(result["reason"], "token-envelope-invalid")
                self.assertEqual(result["queryAttempts"], 0); self.assertEqual(len(fake.calls), 2)
        fake = FakeTransport(); del fake.token["expires_on"]
        self.assertEqual(self.run_probe(fake)[1]["reason"], "token-envelope-invalid")

    def test_wrong_vm_and_token_http_failure_stop_at_exact_stage(self):
        fake = FakeTransport(); fake.metadata = b"wrong"
        _, result, fake, _ = self.run_probe(fake)
        self.assertEqual((result["reason"], len(fake.calls)), ("vm-binding-mismatch", 1))
        self.assertEqual(result["tokenAttempts"], 0)
        fake = FakeTransport(); fake.token_status = 400
        _, result, fake, _ = self.run_probe(fake)
        self.assertEqual((result["reason"], result["tokenHttpStatus"], len(fake.calls)), ("token-http", 400, 2))
        self.assertEqual(result["queryAttempts"], 0)

    def test_config_allowlist_derived_vm_and_attempt_limits_fail_before_http(self):
        for change in ({"extra": "PRIVATE"}, {"vmId": config()["vmId"] + "a"}, {"attempt": 3},
                       {"phase": "after-grant", "attempt": 4}, {"schemaVersion": True},
                       {"container": "../PRIVATE"}, {"notAfterUTC": "2030-01-01T04:31:00Z"}):
            c = dict(config(), **change)
            code, result, fake, _ = self.run_probe(c=c)
            self.assertEqual(code, 2); self.assertEqual(fake.calls, [])
            self.assertEqual(result["queryAttempts"], 0)

    def test_expiry_initially_and_after_each_response_never_retries(self):
        expired = obs.timestamp(config()["notAfterUTC"])
        code, result, fake, _ = self.run_probe(clock=lambda: expired)
        self.assertEqual((code, result["reason"], len(fake.calls)), (2, "deadline-expired", 0))
        for index in (1, 2, 3):
            fake = FakeTransport(); current = [NOW]
            fake.on_call = lambda i: current.__setitem__(0, expired) if i == index else None
            code, result, fake, _ = self.run_probe(fake, clock=lambda: current[0])
            self.assertEqual((code, result["reason"], len(fake.calls)), (2, "deadline-expired", index))
            self.assertEqual(result["queryResponsesReceived"], int(index == 3))

    def test_timeout_attempts_are_conservative_and_do_not_claim_delivery(self):
        for index in (1, 2, 3):
            fake = FakeTransport(); fake.raise_at = index
            code, result, fake, _ = self.run_probe(fake)
            self.assertEqual((code, len(fake.calls)), (2, index))
            self.assertEqual(result["queryAttempts"], int(index == 3))
            self.assertEqual(result["queryResponsesReceived"], 0)
            self.assertNotIn("dataPlaneRequests", result)

    def test_only_specific_same_principal_read_denial_qualifies(self):
        action = "Microsoft.DocumentDB/databaseAccounts/readMetadata"
        for principal, suffix, expected in ((PRINCIPAL, action, "rbac-denied"),
                                             (WF, action, "inconclusive"),
                                             (PRINCIPAL, "Microsoft.DocumentDB/databaseAccounts/write", "inconclusive")):
            fake = FakeTransport(); fake.query_status = 403
            fake.query_body = {"message": f"PRIVATE Request is blocked because principal [{principal}] does not have required RBAC permissions to perform action [{suffix}]"}
            _, result, fake, _ = self.run_probe(fake)
            self.assertEqual(result["classification"], expected); self.assertEqual(len(fake.calls), 3)
        for status in (401, 403, 429, 500):
            fake = FakeTransport(); fake.query_status = status; fake.query_body = {"message": "PRIVATE"}
            self.assertEqual(self.run_probe(fake)[1]["classification"], "inconclusive")

    def test_unexpected_success_is_not_read_proof(self):
        for value in ([], [True], [1, 1], ["PRIVATE"]):
            fake = FakeTransport(); fake.query_body = {"Documents": value}
            self.assertEqual(self.run_probe(fake)[1]["classification"], "inconclusive")

    def denial(self, principal=PRINCIPAL, action=None, article=True):
        action = action or "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery"
        return f"Request is blocked because principal [{principal}] does not have {'the ' if article else ''}required RBAC permissions to perform action [{action}] on resource [PRIVATE-resource]."

    def test_all_supported_message_fields_and_phrase_variants_keep_same_clause_proof(self):
        for field in ("message", "Message", "Errors"):
            for article in (False, True):
                for negation in ("does not", "doesn't"):
                    text = self.denial(article=article).replace("does not", negation)
                    value = {field: [text] if field == "Errors" else text, "ignored": "PRIVATE"}
                    result = obs.classify(403, {"x-ms-substatus": "5302"}, json.dumps(value).encode(), PRINCIPAL)
                    self.assertEqual(result["classification"], "rbac-denied")
                    proof = result["rbacEvidence"]["proof"]
                    self.assertEqual(proof["fieldPath"], "Errors[0]" if field == "Errors" else field)
                    self.assertEqual(proof["servicePrincipalId"], PRINCIPAL)
                    self.assertEqual(proof["readAction"], "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery")
                    self.assertEqual(proof["decodedClauseSHA256"], hashlib.sha256(obs.DENIAL.search(text)[0].encode()).hexdigest())
                    self.assertNotIn("PRIVATE", json.dumps(result))

    def test_expected_identity_or_action_elsewhere_cannot_supply_a_missing_clause_component(self):
        values = [
            {"Errors": [self.denial(WF), "expected principal " + PRINCIPAL]},
            {"message": self.denial(WF) + " expected principal [" + PRINCIPAL + "]"},
            {"message": "principal [" + PRINCIPAL + "]", "Errors": [self.denial(WF)]},
            {"message": self.denial(action="Microsoft.DocumentDB/databaseAccounts/write") + " Microsoft.DocumentDB/databaseAccounts/readMetadata"},
            {"Errors": [self.denial(), self.denial(WF)]},
            {"Errors": [self.denial(), self.denial(action="Microsoft.DocumentDB/databaseAccounts/write")]},
        ]
        for value in values:
            result = obs.classify(403, {"x-ms-substatus": "5302"}, json.dumps(value).encode(), PRINCIPAL)
            self.assertEqual(result["classification"], "inconclusive")
            self.assertNotIn("proof", result["rbacEvidence"])
            self.assertNotIn(WF, json.dumps(result))

    def test_substatus_or_unrecognized_field_never_replaces_required_service_message(self):
        for value in ({}, {"code": "Forbidden"}, {"error": {"message": self.denial()}},
                      {"message": "PRIVATE"}, {"Errors": "PRIVATE"}, {"Errors": [self.denial(), {}]}):
            result = obs.classify(403, {"x-ms-substatus": "5302"}, json.dumps(value).encode(), PRINCIPAL)
            self.assertEqual(result["classification"], "inconclusive")
            self.assertFalse(result["rbacEvidence"]["strictMatch"])
            self.assertNotIn("PRIVATE", json.dumps(result))
        for status in (400, 401, 429, 500):
            self.assertEqual(obs.classify(status, {}, json.dumps({"Errors": [self.denial()]}).encode(), PRINCIPAL)["classification"], "inconclusive")

    def test_parser_rejects_duplicate_fields_or_unbounded_message_arrays(self):
        text = json.dumps(self.denial())
        for body in (('{"message":"PRIVATE","message":' + text + '}').encode(),
                     b'not-json PRIVATE', b'[]', json.dumps({"Errors": [self.denial()] * 9}).encode(),
                     json.dumps({"message": self.denial() * 9}).encode()):
            result = obs.classify(403, {}, body, PRINCIPAL)
            self.assertEqual(result["classification"], "inconclusive")
            self.assertFalse(result["rbacEvidence"]["strictMatch"])
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_new_trial_four_hour_compute_and_96_hour_approval_bounds(self):
        c = config()
        c.update(approvedAtUTC="2029-12-28T05:00:00Z", firstVMIntentUTC="2030-01-01T00:30:00Z",
                 notAfterUTC="2030-01-01T04:30:00Z")
        obs.validate(c, lambda: NOW)
        for change in ({"notAfterUTC": "2030-01-01T04:30:01Z"}, {"approvedAtUTC": "2029-12-28T04:29:59Z"}):
            with self.assertRaises(obs.SafeFailure): obs.validate(dict(c, **change), lambda: NOW)

    def test_bounded_diagnostic_output_even_with_all_supported_fields(self):
        fake = FakeTransport(); fake.query_status = 403
        fake.query_body = {"message": "PRIVATE", "Message": "PRIVATE", "Errors": [self.denial()] * 8}
        code, result, fake, lines = self.run_probe(fake)
        self.assertEqual(code, 0); self.assertEqual(result["classification"], "rbac-denied")
        self.assertEqual(len(fake.calls), 3)

    def test_exact_final_wrapper_live_shape_denial_and_body_proof(self):
        fake = FakeTransport(); fake.query_status = 403; fake.query_body = {"Errors": [self.denial()]}
        code, lines, fake = self.execute_final(fake=fake)
        result = lines[-1]
        self.assertEqual((code, result["classification"], len(fake.calls)), (0, "rbac-denied", 3))
        self.assertEqual(result["rbacEvidence"]["proof"]["fieldPath"], "Errors[0]")
        self.assertEqual(result["rbacEvidence"]["proof"]["servicePrincipalId"], PRINCIPAL)
        self.assertEqual(result["responseSHA256"], hashlib.sha256(json.dumps(fake.query_body).encode()).hexdigest())
        self.assertEqual(json.loads(fake.calls[-1][3]), {"query": "SELECT VALUE 1 FROM c", "parameters": []})
        self.assertEqual({k.lower(): v for k, v in fake.calls[-1][2].items()}["x-ms-max-item-count"], "1")

    def test_exact_final_wrapper_unknown_shape_preserves_safe_diagnosis(self):
        fake = FakeTransport(); fake.query_status = 403; fake.query_body = {"Errors": [{"PRIVATE": "PRIVATE"}]}
        code, lines, fake = self.execute_final(fake=fake)
        self.assertEqual(code, 2); self.assertEqual(len(fake.calls), 3)
        self.assertEqual(lines[-1]["rbacEvidence"]["parseStatus"], "unsupported-message-shape")
        self.assertEqual(lines[-1]["queryResponsesReceived"], 1)

    def execute_final(self, config_value=None, fail_import=False, query_timeout=False, fake=None):
        c = config_value or config()
        script_sha = hashlib.sha256(builder.OBSERVER.read_bytes()).hexdigest()
        with patch.object(builder.observer, "now", return_value=NOW), patch.object(builder.observer, "validate", side_effect=lambda value: obs.validate(value, lambda: NOW)):
            body, _ = builder.body(c, script_sha)
        source = body["properties"]["source"]["script"].split("AF_B06_V4'\n", 1)[1].rsplit("\nAF_B06_V4", 1)[0]
        fake = fake or FakeTransport(c)
        if query_timeout: fake.raise_at = 3
        class Response:
            def __init__(self, status, headers, data): self.status, self.headers, self.data = status, headers, data
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, maximum): return self.data[:maximum]
        class Opener:
            def open(self, request, timeout):
                if timeout != 10: raise AssertionError("Unexpected timeout")
                return Response(*fake(request.get_method(), request.full_url, dict(request.header_items()), request.data))
        def imported(code, namespace):
            if fail_import: raise RuntimeError("PRIVATE module initialization")
            builtins.exec(code, namespace)
        class Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None): return NOW
        output = io.StringIO()
        with patch("sys.platform", "linux"), patch("os.geteuid", return_value=0), \
             patch("datetime.datetime", Clock), patch("urllib.request.build_opener", return_value=Opener()), \
             patch("os.open", side_effect=AssertionError("Real file access forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("Real process forbidden")), \
             patch("socket.create_connection", side_effect=AssertionError("Real network forbidden")), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                builtins.exec(compile(source, "exact-b06-v4-final-wrapper", "exec"), {"exec": imported})
        self.assertNotIn("PRIVATE", output.getvalue()); self.assertLessEqual(len(output.getvalue().encode()), 4096)
        return stopped.exception.code, [json.loads(line) for line in output.getvalue().splitlines()], fake

    def test_exact_final_wrapper_both_phases_opaque_success(self):
        for phase in ("before-grant", "after-grant"):
            c = dict(config(), phase=phase)
            code, lines, fake = self.execute_final(c)
            self.assertEqual((code, lines[-1]["classification"], len(fake.calls)), (0, "read-succeeded", 3))
            self.assertEqual(lines[-1]["phase"], phase)

    def test_exact_final_wrapper_query_timeout_has_intent_without_response(self):
        code, lines, fake = self.execute_final(query_timeout=True)
        self.assertEqual(code, 2); self.assertEqual(len(fake.calls), 3)
        self.assertEqual(lines[-1]["reason"], "transport-timeout")
        self.assertEqual((lines[-1]["queryAttempts"], lines[-1]["queryResponsesReceived"]), (1, 0))

    def test_exact_final_wrapper_import_failure_reports_unknown_counts(self):
        code, lines, fake = self.execute_final(fail_import=True)
        self.assertEqual(code, 2); self.assertEqual(fake.calls, [])
        self.assertFalse(lines[-1]["requestCountsKnown"])
        self.assertEqual(lines[-1]["reason"], "runtime-unavailable")

    def test_builder_requires_exact_identity_and_never_overwrites(self):
        binding = {k: v for k, v in config().items() if k not in ("phase", "attempt")}
        receipt = {"id": binding["vmId"], "observedAtUTC": binding["vmIdentityObservedAtUTC"],
                   "responseSHA256": binding["vmIdentityReceiptSHA256"],
                   "identity": {"type": "SystemAssigned", "principalId": PRINCIPAL, "tenantId": WF}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); bp = root / "binding.json"; rp = root / "identity.json"
            bp.write_text(json.dumps(binding)); rp.write_text(json.dumps(receipt))
            args = (bp, builder.sha(bp.read_bytes()), rp, builder.sha(rp.read_bytes()), builder.sha(builder.OBSERVER.read_bytes()), root / "prepared")
            with patch.object(builder.observer, "validate", side_effect=lambda value: obs.validate(value, lambda: NOW)):
                result = builder.prepare(*args)
                self.assertEqual((result["maximumBefore"], result["maximumAfter"]), (2, 3))
                self.assertEqual(len(result["artifacts"]), 5); self.assertFalse(result["executed"])
                with self.assertRaises(FileExistsError): builder.prepare(*args)
                receipt["identity"]["principalId"] = WF; rp.write_text(json.dumps(receipt))
                with self.assertRaises(builder.observer.SafeFailure):
                    builder.prepare(bp, builder.sha(bp.read_bytes()), rp, builder.sha(rp.read_bytes()), args[4], root / "bad")
                self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
