import base64
import contextlib
import copy
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_b06", Path(__file__).with_name("prepare-b06-observer.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
NOW = dt.datetime(2026, 9, 24, 0, 30, tzinfo=dt.timezone.utc)
FIRST = "2026-09-24T00:20:00Z"
RECEIPT = {"observedAtUTC": "2026-09-24T00:25:00Z", "responseSHA256": "a" * 64,
           "id": m.VM, "identity": {"type": "SystemAssigned",
           "principalId": "11111111-1111-1111-1111-111111111111",
           "tenantId": "22222222-2222-2222-2222-222222222222"}}


class PreparationTests(unittest.TestCase):
    def test_identity_and_receipt_causality_refused(self):
        for mutate in (lambda r: r.update(id=m.ACCOUNT),
                       lambda r: r.update(observedAtUTC="2026-09-24T00:19:59Z"),
                       lambda r: r.update(observedAtUTC="2026-09-24T00:31:00Z"),
                       lambda r: r["identity"].update(type="SystemAssigned, UserAssigned"),
                       lambda r: r["identity"].update(principalId="")):
            with self.subTest(mutate=mutate):
                receipt = copy.deepcopy(RECEIPT)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    m.bindings(receipt, FIRST, NOW)

    def test_deadline_minimum_and_expiry(self):
        result = m.bindings(RECEIPT, FIRST, NOW)
        self.assertEqual(result["notAfterUTC"], "2026-09-24T01:50:00+00:00")
        with self.assertRaises(ValueError):
            m.bindings(RECEIPT, FIRST, m.timestamp(result["notAfterUTC"]))
        later = copy.deepcopy(RECEIPT)
        later["observedAtUTC"] = "2026-09-24T03:30:00Z"
        result = m.bindings(later, "2026-09-24T03:29:00Z", m.timestamp(later["observedAtUTC"]))
        self.assertEqual(result["notAfterUTC"], "2026-09-24T04:18:14+00:00")

    def test_only_four_exact_ids_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = root / "receipt.json"
            raw = json.dumps(RECEIPT).encode()
            receipt.write_bytes(raw)
            result = m.prepare(receipt, m.digest(raw), FIRST, root / "prepared", NOW)
            names = [x["id"].rsplit("/", 1)[1] for x in result["artifacts"]]
            self.assertEqual(names, ["af-b06-observe-before-01", "af-b06-observe-after-01", "af-b06-observe-after-02", "af-b06-observe-after-03"])
            self.assertFalse(result["executed"])
            for item in result["artifacts"]:
                self.assertEqual(m.digest((root / "prepared" / item["bodyFile"]).read_bytes()), item["bodySHA256"])
            with self.assertRaises(FileExistsError):
                m.prepare(receipt, m.digest(raw), FIRST, root / "prepared", NOW)

    def test_tampered_observer_refused(self):
        with self.assertRaises(ValueError):
            m.body({**m.bindings(RECEIPT, FIRST, NOW), "phase": "before-grant"}, m.OBSERVER.read_bytes() + b"\n")

    def test_seal_mismatch_writes_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(RECEIPT))
            with self.assertRaises(ValueError):
                m.prepare(receipt, "b" * 64, FIRST, root / "prepared", NOW)
            self.assertFalse((root / "prepared").exists())

    def run_wrapper(self, failing=False, safe_failure=False, script=None, config=None):
        config = config or {**m.bindings(RECEIPT, FIRST, NOW), "phase": "before-grant"}
        body, _ = m.body(config, m.OBSERVER.read_bytes())
        source = (script or body["properties"]["source"]["script"]).split("AF_B06_OBSERVER'\n", 1)[1].rsplit("\nAF_B06_OBSERVER", 1)[0]
        calls = []
        claims = {"oid": config["principalId"], "tid": config["tenantId"], "xms_mirid": m.VM,
                  "aud": "https://cosmos.azure.com", "exp": 9999999999}
        token = "header." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".private-signature"
        class Response:
            status = 200
            headers = {}
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, maximum): return json.dumps(self.value).encode()
        class Opener:
            def open(self, request, timeout):
                calls.append(request.get_method())
                if failing:
                    raise RuntimeError("secret-token-should-never-print")
                if safe_failure:
                    return Response({"access_token": "intentionally-invalid-token"})
                return Response({"access_token": token} if len(calls) == 1 else {"Documents": [1]})
        class Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None): return NOW
        output = io.StringIO()
        with patch("urllib.request.build_opener", return_value=Opener()), patch("datetime.datetime", Clock), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                exec(compile(source, "prepared-wrapper", "exec"), {})
        self.assertNotIn(token, output.getvalue())
        self.assertNotIn("secret-token", output.getvalue())
        return calls, stopped.exception.code, json.loads(output.getvalue())

    def test_wrapper_uses_single_request_pair_and_retains_seals(self):
        calls, status, result = self.run_wrapper()
        self.assertEqual(calls, ["GET", "POST"])
        self.assertEqual(status, 0)
        self.assertEqual(result["classification"], "read-succeeded")
        self.assertFalse(result["guiAssessment"])
        self.assertEqual(result["observerSHA256"], m.OBSERVER_SHA)

    def test_wrapper_redacts_unexpected_errors_without_retry(self):
        calls, status, result = self.run_wrapper(True)
        self.assertEqual(calls, ["GET"])
        self.assertEqual(status, 2)
        self.assertEqual(result, {"classification": "inconclusive", "reason": "observer-failed"})

    def test_wrapper_preserves_allowlisted_safe_failure(self):
        calls, status, result = self.run_wrapper(safe_failure=True)
        self.assertEqual(calls, ["GET"])
        self.assertEqual(status, 2)
        self.assertEqual(result, {"classification": "inconclusive", "reason": "managed-identity-binding-mismatch"})

    def test_supplemental_preparation_is_explicit_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); receipt = root / "receipt.json"
            raw = json.dumps(RECEIPT).encode(); receipt.write_bytes(raw)
            result = m.prepare(receipt, m.digest(raw), FIRST, root / "prepared", NOW, supplemental=True)
            self.assertTrue(result["supplementalBeforeRequiresNewApproval"])
            self.assertEqual([a["id"].rsplit("/", 1)[1] for a in result["artifacts"]], ["af-b06-observe-before-02", "af-b06-observe-after-01"])
            for item in result["artifacts"]:
                script = json.loads((root / "prepared" / item["bodyFile"]).read_bytes())["properties"]["source"]["script"]
                config = json.loads((root / "prepared" / item["configFile"]).read_bytes())
                calls, status, observed = self.run_wrapper(script=script, config=config)
                self.assertEqual((calls, status, observed["classification"]), (["GET", "POST"], 0, "read-succeeded"))


if __name__ == "__main__":
    unittest.main()
