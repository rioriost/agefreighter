#!/usr/bin/env python3
"""Bounded same-VM Cosmos read observation; importing performs no I/O.

Tokens remain opaque and in memory. Only a reviewed body may call run_once.
This module has no launcher, grant, retry, continuation or source-write path.
"""
import datetime as dt
import email.utils
import hashlib
import json
import os
import re
import signal
import sys
import urllib.error
import urllib.parse
import urllib.request

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
SHA = r"[0-9a-f]{64}"
QUERY = {"query": "SELECT TOP 1 VALUE 1 FROM c", "parameters": []}
RESOURCE = "https://cosmos.azure.com"
IMDS_VM = "http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text"
IMDS_TOKEN = "http://169.254.169.254/metadata/identity/oauth2/token"
FIELDS = {"schemaVersion", "trialId", "phase", "attempt", "vmId", "principalId", "tenantId",
          "accountId", "database", "container", "approvedAtUTC", "firstVMIntentUTC", "notAfterUTC",
          "vmIdentityObservedAtUTC", "vmIdentityReceiptSHA256"}
BINDING_FIELDS = ("trialId", "vmId", "principalId", "tenantId", "accountId", "database", "container")
COUNTERS = ("vmMetadataAttempts", "vmMetadataResponsesReceived", "tokenAttempts", "tokenResponsesReceived",
            "queryAttempts", "queryResponsesReceived")
READ_ACTIONS = {"Microsoft.DocumentDB/databaseAccounts/readMetadata",
                "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery",
                "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read"}
REASONS = {"config-invalid", "deadline-expired", "vm-metadata-http", "vm-binding-mismatch",
           "token-http", "token-envelope-invalid", "transport-unavailable", "transport-timeout",
           "redirect-refused", "response-limit", "query-unclassified", "runtime-unavailable",
           "internal-error", "output-limit"}


class SafeFailure(Exception):
    pass


def require(value, reason="config-invalid"):
    if not value:
        raise SafeFailure(reason)


def now():
    return dt.datetime.now(dt.timezone.utc)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def timestamp(value):
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.utcoffset() == dt.timedelta(0))
        return result
    except (ValueError, TypeError, AttributeError):
        raise SafeFailure("config-invalid") from None


def validate(config, clock=now):
    require(isinstance(config, dict) and set(config) == FIELDS)
    require(type(config["schemaVersion"]) is int and config["schemaVersion"] == 2)
    require(config["phase"] in ("before-grant", "after-grant") and type(config["attempt"]) is int)
    require(1 <= config["attempt"] <= 2 if config["phase"] == "before-grant" else 1 <= config["attempt"] <= 3)
    for key in ("trialId", "principalId", "tenantId"):
        require(isinstance(config[key], str) and re.fullmatch(UUID, config[key]))
    require(isinstance(config["vmIdentityReceiptSHA256"], str) and re.fullmatch(SHA, config["vmIdentityReceiptSHA256"]))
    prefix = rf"/subscriptions/({UUID})/resourceGroups/([A-Za-z0-9_.()-]{{1,90}})/providers/"
    vm = re.fullmatch(prefix + r"Microsoft.Compute/virtualMachines/af-([a-f0-9]{20})", config["vmId"] if isinstance(config["vmId"], str) else "")
    account = re.fullmatch(prefix + r"Microsoft.DocumentDB/databaseAccounts/([a-z0-9-]{3,44})", config["accountId"] if isinstance(config["accountId"], str) else "")
    require(vm and account and vm.group(1) == account.group(1))
    require(vm.group(3) == config["trialId"].replace("-", "")[:20])
    for key in ("database", "container"):
        require(isinstance(config[key], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", config[key]))
    approved, first, expiry, observed = [timestamp(config[k]) for k in
        ("approvedAtUTC", "firstVMIntentUTC", "notAfterUTC", "vmIdentityObservedAtUTC")]
    require(approved <= first <= observed < expiry and expiry - first <= dt.timedelta(minutes=90)
            and expiry - approved <= dt.timedelta(hours=4))
    require(observed <= clock() < expiry, "deadline-expired")
    return "https://" + account.group(3) + ".documents.azure.com/dbs/" + config["database"] + "/colls/" + config["container"] + "/docs"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise SafeFailure("redirect-refused")


def transport(method, url, headers, body=None, maximum=65536):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = opener.open(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    except TimeoutError:
        raise SafeFailure("transport-timeout") from None
    except (urllib.error.URLError, OSError):
        raise SafeFailure("transport-unavailable") from None
    with response:
        data = response.read(maximum + 1)
        require(len(data) <= maximum, "response-limit")
        return response.status, dict(response.headers.items()), data


def opaque_token(body, clock):
    try:
        envelope = json.loads(body)
        require(isinstance(envelope, dict), "token-envelope-invalid")
        token = envelope.get("access_token")
        require(isinstance(token, str) and 0 < len(token.encode()) <= 32768
                and all(ord(c) >= 33 and ord(c) != 127 for c in token), "token-envelope-invalid")
        require(str(envelope.get("token_type", "")).lower() == "bearer"
                and envelope.get("resource", "").rstrip("/") == RESOURCE, "token-envelope-invalid")
        expires = envelope.get("expires_on")
        require(type(expires) in (str, int) and re.fullmatch(r"[0-9]{1,12}", str(expires)), "token-envelope-invalid")
        require(int(expires) > clock().timestamp() + 30, "token-envelope-invalid")
        if "not_before" in envelope:
            starts = envelope["not_before"]
            require(type(starts) in (str, int) and re.fullmatch(r"[0-9]{1,12}", str(starts))
                    and int(starts) <= clock().timestamp() + 30, "token-envelope-invalid")
        return token
    except (ValueError, TypeError, AttributeError, UnicodeError):
        raise SafeFailure("token-envelope-invalid") from None


def classify(status, headers, body, principal):
    result = {"classification": "inconclusive", "httpStatus": status,
              "responseSHA256": hashlib.sha256(body).hexdigest(), "responseBytes": len(body)}
    lower = {key.lower(): value for key, value in headers.items()}
    for key, target, pattern in (("x-ms-activity-id", "activityId", UUID),
                                 ("x-ms-substatus", "substatus", r"[0-9]{1,8}"),
                                 ("x-ms-request-charge", "requestCharge", r"[0-9]{1,12}(?:\.[0-9]{1,12})?")):
        value = lower.get(key, "")
        if isinstance(value, str) and re.fullmatch(pattern, value, re.IGNORECASE):
            result[target] = value
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            return result
        if status == 200 and value.get("Documents") == [1] and type(value["Documents"][0]) is int:
            result.update(classification="read-succeeded", constantRows=1)
        elif status == 403:
            message = value.get("message", "")
            if not isinstance(message, str):
                return result
            # Retain only the decision booleans, never arbitrary action/body text.
            action = READ_ACTIONS.intersection(re.findall(r"Microsoft\.DocumentDB/databaseAccounts/[a-zA-Z/]+", message))
            rbac = re.search(r"(?:does not|doesn't) have required RBAC permissions", message, re.IGNORECASE)
            same = re.search(r"(?<![0-9a-f])" + re.escape(principal) + r"(?![0-9a-f])", message, re.IGNORECASE)
            if action and rbac and same:
                result.update(classification="rbac-denied", servicePrincipalMatches=True, deniedReadActionPresent=True)
    except (ValueError, TypeError):
        pass
    return result


def stdout(value):
    print(encoded(value).decode(), flush=True)


def run_once(config, config_sha, observer_sha, request=transport, emit=stdout, clock=now):
    counters = dict.fromkeys(COUNTERS, 0)
    stage, output_bytes, validated = "initialization", 0, False
    checks = {"configValidated": False, "vmMatches": False, "tokenEnvelopeValid": False}
    http = {}
    def publish(value):
        nonlocal output_bytes
        raw = encoded(value)
        require(output_bytes + len(raw) + 1 <= 4096, "output-limit")
        output_bytes += len(raw) + 1
        emit(value)
    def event(next_stage):
        nonlocal stage
        stage = next_stage
        publish({"schemaVersion": 2, "event": "stage", "stage": stage, **counters})
    def invoke(kind, method, url, headers, body=None, maximum=65536):
        nonlocal stage
        stage = {"vmMetadata": "vm-metadata", "token": "token", "query": "query"}[kind]
        validate(config, clock)
        counters[kind + "Attempts"] += 1
        event(stage)
        status, headers_out, raw = request(method, url, headers, body, maximum)
        require(type(status) is int and 100 <= status <= 599 and isinstance(headers_out, dict)
                and isinstance(raw, bytes), "internal-error")
        require(len(raw) <= maximum, "response-limit")
        counters[kind + "ResponsesReceived"] += 1
        http[kind + "HttpStatus"] = status
        validate(config, clock)
        return status, headers_out, raw
    def timed_out(_signum, _frame):
        raise SafeFailure("transport-timeout")
    result = {"classification": "inconclusive"}
    reason = None
    previous_handler = None
    armed = False
    try:
        require(sys.platform == "linux" and os.geteuid() == 0, "runtime-unavailable")
        previous_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(55)
        armed = True
        event("config")
        endpoint = validate(config, clock)
        require(re.fullmatch(SHA, config_sha or "") and re.fullmatch(SHA, observer_sha or ""))
        checks["configValidated"] = validated = True
        status, _, raw = invoke("vmMetadata", "GET", IMDS_VM, {"Metadata": "true"}, maximum=2048)
        require(status == 200, "vm-metadata-http")
        try:
            matches = raw.decode("utf-8").strip().lower() == config["vmId"].lower()
        except UnicodeError:
            matches = False
        require(matches, "vm-binding-mismatch")
        checks["vmMatches"] = True
        token_url = IMDS_TOKEN + "?" + urllib.parse.urlencode({"api-version": "2018-02-01", "resource": RESOURCE, "object_id": config["principalId"]})
        status, _, raw = invoke("token", "GET", token_url, {"Metadata": "true"})
        require(status == 200, "token-http")
        token = opaque_token(raw, clock)
        checks["tokenEnvelopeValid"] = True
        headers = {"Authorization": urllib.parse.quote("type=aad&ver=1.0&sig=" + token, safe=""),
                   "x-ms-date": email.utils.format_datetime(clock(), usegmt=True), "x-ms-version": "2018-12-31",
                   "x-ms-documentdb-isquery": "True", "x-ms-documentdb-query-enablecrosspartition": "True",
                   "x-ms-max-item-count": "1", "Content-Type": "application/query+json", "Accept": "application/json"}
        status, response_headers, raw = invoke("query", "POST", endpoint, headers, encoded(QUERY))
        stage = "classification"
        result = classify(status, response_headers, raw, config["principalId"])
        reason = "query-unclassified" if result["classification"] == "inconclusive" else None
    except SafeFailure as error:
        reason = str(error) if str(error) in REASONS else "internal-error"
    except Exception:
        reason = "internal-error"
    finally:
        if armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    final = {"schemaVersion": 2, "event": "result", "stage": stage, "guiAssessment": False,
             "evidenceLayer": "standalone-same-vm-managed-identity-data-plane-observer",
             "observedAtUTC": clock().isoformat(), "attemptsArePreDispatchIntents": True,
             "responsesAreCompleteBoundedResponses": True, "automaticRetries": 0, "continuationFollowed": False,
             **counters, **checks, **http, **result}
    if reason:
        final["reason"] = reason
    if validated:
        final.update(trialId=config["trialId"], phase=config["phase"], attempt=config["attempt"],
                     configSHA256=config_sha, observerSHA256=observer_sha,
                     bindingSHA256=digest({key: config[key] for key in BINDING_FIELDS}), querySHA256=digest(QUERY),
                     identityEvidence="fresh-arm-system-identity+guest-imds-resource-id+selected-imds-object-id",
                     tokenHandling="opaque; Cosmos validates presented token")
    try:
        publish(final)
    except Exception:
        # The wrapper emits its fixed unknown-state result if serialization or
        # output itself fails. Never infer zero requests from that fallback.
        raise SafeFailure("output-limit") from None
    return 0 if result["classification"] in ("rbac-denied", "read-succeeded") else 2


if __name__ == "__main__":
    print('{"preparedOnly":true,"executed":false,"requiresReviewedBoundBody":true}')
    raise SystemExit(2)
