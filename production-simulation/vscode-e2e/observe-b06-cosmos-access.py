#!/usr/bin/env python3
"""One explicitly approved, read-only Cosmos request from the dedicated VM.

No Azure CLI, keys, credential files, writes, pagination, retry or token output.
Run only on the newly approved trial VM; preparation/tests do not run this probe.
The token lives in memory. Only constant projected values can reach the receipt.
"""
import argparse
import base64
import datetime as dt
import email.utils
import hashlib
import json
import re
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
QUERY = {"query": "SELECT TOP 1 VALUE 1 FROM c", "parameters": []}
IMDS = "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fcosmos.azure.com"
MAX_BYTES = 65536
FIELDS = {"schemaVersion", "trialId", "phase", "vmId", "principalId", "tenantId", "accountId", "database", "container", "approvedAtUTC", "notAfterUTC"}


class SafeFailure(Exception):
    """An allowlisted reason, never an upstream exception or response body."""


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp(value):
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() != dt.timedelta(0):
            raise ValueError()
        return parsed
    except (ValueError, AttributeError, TypeError):
        raise SafeFailure("invalid-approval-window") from None


def validate(config, now=None):
    now = now or utcnow()
    if not isinstance(config, dict) or set(config) != FIELDS or type(config.get("schemaVersion")) is not int or config["schemaVersion"] != 1:
        raise SafeFailure("invalid-config-fields")
    for key in ("trialId", "principalId", "tenantId"):
        if not isinstance(config[key], str) or not re.fullmatch(UUID, config[key]):
            raise SafeFailure("invalid-identity-binding")
    if config["phase"] not in ("before-grant", "after-grant"):
        raise SafeFailure("invalid-phase")
    resource = rf"/subscriptions/{UUID}/resourceGroups/[a-zA-Z0-9_.()-]{{1,90}}/providers/"
    if not isinstance(config["vmId"], str) or not re.fullmatch(resource + r"Microsoft.Compute/virtualMachines/[a-zA-Z0-9_-]{1,64}", config["vmId"]):
        raise SafeFailure("invalid-vm-binding")
    match = re.fullmatch(resource + r"Microsoft.DocumentDB/databaseAccounts/([a-z0-9-]{3,44})", config["accountId"] if isinstance(config["accountId"], str) else "")
    if not match or config["vmId"].split("/")[2] != config["accountId"].split("/")[2]:
        raise SafeFailure("invalid-account-binding")
    for key in ("database", "container"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", config[key]):
            raise SafeFailure("invalid-source-name")
    approved, expiry = stamp(config["approvedAtUTC"]), stamp(config["notAfterUTC"])
    if not approved <= now < expiry or not dt.timedelta(0) < expiry - approved <= dt.timedelta(hours=4):
        raise SafeFailure("approval-window-expired-or-unbounded")
    return "https://" + match.group(1) + ".documents.azure.com/dbs/" + config["database"] + "/colls/" + config["container"] + "/docs"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SafeFailure("redirect-refused")


def transport(method, url, headers, body=None):
    # Explicitly ignore proxy environment variables for both IMDS and Cosmos.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = opener.open(request, timeout=20)
    except urllib.error.HTTPError as error:
        response = error
    except (urllib.error.URLError, TimeoutError, OSError):
        raise SafeFailure("transport-unavailable") from None
    with response:
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise SafeFailure("response-limit")
        return response.status, dict(response.headers.items()), data


def token_for(config, request):
    status, _, body = request("GET", IMDS, {"Metadata": "true"})
    if status != 200:
        raise SafeFailure("managed-identity-unavailable")
    try:
        value = json.loads(body)
        token = value["access_token"]
        if not isinstance(token, str) or len(token) > 16384 or len(token.split(".")) != 3:
            raise ValueError()
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        # These are locally inspected token claims, not independent JWT signature
        # validation. The real Cosmos service validates the presented token.
        if str(claims.get("oid", "")).lower() != config["principalId"] or str(claims.get("tid", "")).lower() != config["tenantId"]:
            raise ValueError()
        if str(claims.get("xms_mirid", "")).lower() != config["vmId"].lower() or claims.get("aud", "").rstrip("/") != "https://cosmos.azure.com":
            raise ValueError()
        if float(claims["exp"]) <= time.time() + 30:
            raise ValueError()
    except (KeyError, ValueError, TypeError, AttributeError):
        raise SafeFailure("managed-identity-binding-mismatch") from None
    return token


def summarize(status, headers, body, principal):
    headers = {key.lower(): value for key, value in headers.items()}
    result = {"httpStatus": status, "responseSHA256": hashlib.sha256(body).hexdigest(), "responseBytes": len(body), "classification": "inconclusive"}
    for name, target, pattern in (("x-ms-activity-id", "activityId", UUID), ("x-ms-substatus", "substatus", r"[0-9]{1,8}"), ("x-ms-request-charge", "requestCharge", r"[0-9]{1,12}(?:\.[0-9]{1,12})?")):
        value = headers.get(name, "")
        if re.fullmatch(pattern, value, re.IGNORECASE):
            result[target] = value
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            return result
        if status == 200 and value.get("Documents") == [1] and type(value["Documents"][0]) is int:
            result.update(classification="read-succeeded", constantRows=1)
        elif status == 403:
            message = str(value.get("message", ""))
            actions = sorted(set(re.findall(r"Microsoft\.DocumentDB/databaseAccounts/[a-zA-Z/]+", message)))
            # 403 alone can mean networking or policy. Require the service's
            # explicit RBAC diagnosis bound to the expected principal.
            rbac = re.search(r"(?:does not|doesn't) have required RBAC permissions", message, re.IGNORECASE)
            principal_match = re.search(r"(?<![0-9a-f])" + re.escape(principal) + r"(?![0-9a-f])", message, re.IGNORECASE)
            if rbac and principal_match and actions:
                result.update(classification="rbac-denied", deniedActions=actions, servicePrincipalMatches=True)
    except (ValueError, TypeError):
        pass
    return result


def observe(config, request=transport):
    endpoint = validate(config)
    token = token_for(config, request)
    validate(config)  # Do not send a data-plane request after approval expires.
    headers = {"Authorization": urllib.parse.quote("type=aad&ver=1.0&sig=" + token, safe=""),
               "x-ms-date": email.utils.format_datetime(utcnow(), usegmt=True), "x-ms-version": "2018-12-31",
               "x-ms-documentdb-isquery": "True", "x-ms-documentdb-query-enablecrosspartition": "True",
               "x-ms-max-item-count": "1", "Content-Type": "application/query+json", "Accept": "application/json"}
    status, response_headers, body = request("POST", endpoint, headers, json.dumps(QUERY).encode())
    binding = {key: config[key] for key in ("trialId", "vmId", "principalId", "tenantId", "accountId", "database", "container")}
    return {"schemaVersion": 1, "evidenceLayer": "standalone-same-vm-managed-identity-data-plane-observer", "guiAssessment": False,
            "observedAtUTC": utcnow().isoformat(), "binding": binding, "bindingSHA256": digest(binding), "phase": config["phase"],
            "querySHA256": digest(QUERY), "dataPlaneRequests": 1, "automaticRetries": 0, "continuationFollowed": False,
            "claimValidation": "local-binding-only; Cosmos validates presented token", **summarize(status, response_headers, body, config["principalId"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--execute-approved", action="store_true", required=True)
    args = parser.parse_args()
    def timed_out(_signum, _frame):
        raise SafeFailure("total-time-limit")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(60)
    try:
        observer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw = Path(args.config).read_bytes()
        config_hash = hashlib.sha256(raw).hexdigest()
        if len(raw) > 8192 or observer_hash != args.observer_sha256 or config_hash != args.config_sha256:
            raise SafeFailure("reviewed-seal-mismatch")
        result = observe(json.loads(raw))
        result.update(observerSHA256=observer_hash, configSHA256=config_hash)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["classification"] in ("rbac-denied", "read-succeeded") else 2
    except SafeFailure as error:
        print(json.dumps({"classification": "inconclusive", "reason": str(error)}))
        return 2
    except Exception:
        # No exception text, URL, authorization header, token, source body or
        # traceback is safe to serialize from an unexpected transport failure.
        print(json.dumps({"classification": "inconclusive", "reason": "observer-failed"}))
        return 2
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
