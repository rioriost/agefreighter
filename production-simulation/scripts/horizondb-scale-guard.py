"""Published with a reviewed, literal main(settings) call by the operator.

Runs independently of the loaders in Azure Automation. No Azure SDK/packages,
shell commands, passwords, webhooks or worker-supplied deletion paths are used.
"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import hashlib
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ARM = "https://management.azure.com"
HORIZON_API = "2026-10-01-preview"
RESOURCE_API = "2025-04-01"
COMPUTE_API = "2026-04-01"
DISK_API = "2026-10-02"
SNAPSHOT_API = "2026-10-02"
UTC = dt.timezone.utc


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("UTC timestamps are required")
    return parsed


def validate(settings):
    subscription = str(uuid.UUID(settings["subscriptionId"]))
    date = settings["date"]
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError("invalid campaign date")
    dt.datetime.strptime(date, "%Y%m%d")
    base = "/subscriptions/" + subscription + "/resourceGroups/"
    preserved = base + "rg-agefreighter-horizon-tiny-" + date
    disposable = base + "rg-agefreighter-horizon-scale-" + date
    clusters = [
        preserved + "/providers/Microsoft.HorizonDb/clusters/afht-" + date + "-pg" + major
        for major in ("17", "18")
    ]
    approved = {p.lower() for p in [disposable] + clusters}
    targets = settings["deleteTargets"]
    if not targets or len({p.lower() for p in targets}) != len(targets):
        raise ValueError("empty or duplicate deletion scope")
    if any(p.lower() not in approved for p in targets):
        raise ValueError("deletion scope exceeds the approved resource IDs")
    disk = settings["preservedEvidenceDiskId"]
    disk_prefix = preserved + "/providers/Microsoft.Compute/disks/"
    if not disk.lower().startswith(disk_prefix.lower()) or not re.fullmatch(
        r"[a-zA-Z0-9_.-]+", disk[len(disk_prefix):]
    ):
        raise ValueError("evidence must be on the excluded existing managed disk")
    started, deadline = timestamp(settings["startedAtUtc"]), timestamp(settings["deadlineUtc"])
    if not dt.timedelta(0) < deadline - started <= dt.timedelta(hours=24):
        raise ValueError("each enrolled execution lease must be positive and at most 24 hours")
    for key in ("baselineCostUSD", "lagReserveUSD", "deletionReserveUSD", "maxHourlyUSD"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 3750:
            raise ValueError("invalid conservative cost input: " + key)
    if settings["maxHourlyUSD"] <= 0 or settings["lagReserveUSD"] < 100:
        raise ValueError("positive accrual and at least USD100 posting-lag reserve required")
    if settings["deletionReserveUSD"] < settings["maxHourlyUSD"] * 1.5:
        raise ValueError("at least 90 minutes of bounded cleanup cost must be reserved")
    worst = (settings["baselineCostUSD"] + settings["lagReserveUSD"]
             + settings["deletionReserveUSD"]
             + settings["maxHourlyUSD"] * (deadline - started).total_seconds() / 3600)
    if worst > 3750:
        raise ValueError("the complete lease would exceed the monthly USD3750 envelope")
    return preserved, disposable


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("redirect refused for an authenticated request")


class Client:
    def __init__(self):
        self.http = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )
        self.token = None
        self.expires = 0

    def authenticate(self):
        if self.token and time.time() + 120 < self.expires:
            return
        endpoint = os.environ["IDENTITY_ENDPOINT"]
        parsed = urllib.parse.urlsplit(endpoint)
        local = parsed.hostname == "localhost"
        if not local:
            address = ipaddress.ip_address(parsed.hostname)
            local = address.is_loopback or address.is_link_local
        if parsed.scheme != "http" or not local:
            raise RuntimeError("unexpected Automation identity endpoint")
        query = urllib.parse.urlencode({"resource": ARM + "/"})
        request = urllib.request.Request(
            endpoint + ("&" if parsed.query else "?") + query,
            headers={"X-IDENTITY-HEADER": os.environ["IDENTITY_HEADER"], "Metadata": "True"},
        )
        with self.http.open(request, timeout=30) as response:
            token = json.load(response)
        self.token = token["access_token"]
        self.expires = float(token["expires_on"])

    def request(self, method, path, body=None, missing_ok=False):
        url = path if path.startswith("https://") else ARM + path
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "management.azure.com":
            raise RuntimeError("unexpected authenticated request origin")
        self.authenticate()
        for attempt in range(5):
            request = urllib.request.Request(
                url, method=method,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            )
            try:
                with self.http.open(request, timeout=60) as response:
                    raw = response.read()
                    return (json.loads(raw) if raw else {}), response.headers
            except urllib.error.HTTPError as error:
                if error.code == 404 and missing_ok:
                    return None, {}
                if error.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                    wait = min(60, max(2 ** attempt, int(error.headers.get("Retry-After", "0"))))
                    print(json.dumps({"retry": attempt + 1, "http": error.code, "waitSeconds": wait}), flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError("ARM HTTP " + str(error.code) + ": " + error.read().decode()) from error
        raise RuntimeError("ARM retry budget exhausted")

    def operation(self, method, path, body=None):
        result, headers = self.request(method, path, body, missing_ok=method == "DELETE")
        operation = headers.get("Azure-AsyncOperation") or headers.get("Location")
        end = time.monotonic() + 1200
        while operation:
            if time.monotonic() >= end:
                raise RuntimeError("ARM lifecycle operation exceeded 20 minutes")
            time.sleep(10)
            result, _ = self.request("GET", operation)
            state = result.get("status", result.get("properties", {}).get("provisioningState"))
            if state == "Succeeded":
                break
            if state not in ("Accepted", "InProgress", "Running", "Updating", "Deleting", "Creating"):
                raise RuntimeError("ARM lifecycle operation did not succeed: " + json.dumps(result))
        return result


def list_resources(client, group):
    path = group + "/resources?api-version=" + RESOURCE_API
    result = []
    while path:
        page, _ = client.request("GET", path)
        result.extend(page["value"])
        path = page.get("nextLink")
    return result


def preserve_worker_evidence(client, settings, preserved, disposable):
    disk, _ = client.request(
        "GET", settings["preservedEvidenceDiskId"] + "?api-version=" + DISK_API
    )
    if disk["properties"]["provisioningState"] != "Succeeded":
        raise RuntimeError("excluded evidence disk is not durably available")
    if not any(t.lower() == disposable.lower() for t in settings["deleteTargets"]):
        return []
    group, _ = client.request("GET", disposable + "?api-version=" + RESOURCE_API, missing_ok=True)
    if group is None:
        return []
    if group.get("tags", {}).get("campaign") != "afhs-" + settings["date"]:
        raise RuntimeError("disposable group ownership differs")
    workers = []
    for resource in list_resources(client, disposable):
        if resource["type"].lower() != "microsoft.compute/virtualmachines":
            continue
        vm, _ = client.request("GET", resource["id"] + "?api-version=" + COMPUTE_API)
        if vm.get("tags", {}).get("campaign") != "afhs-" + settings["date"]:
            raise RuntimeError("worker ownership differs")
        os_disk = vm["properties"]["storageProfile"]["osDisk"]
        if os_disk.get("diffDiskSettings", {}).get("option") == "Local":
            raise RuntimeError("worker evidence cannot be on an ephemeral OS disk")
        disk_id = os_disk["managedDisk"]["id"]
        if not disk_id.lower().startswith(disposable.lower() + "/providers/microsoft.compute/disks/"):
            raise RuntimeError("worker OS disk is outside the disposable owned group")
        disk, _ = client.request("GET", disk_id + "?api-version=" + DISK_API)
        if disk["properties"]["diskSizeGB"] > 128:
            raise RuntimeError("worker evidence snapshot exceeds the reviewed 128GiB bound")
        workers.append((resource["id"], disk_id))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: client.operation(
            "POST", item[0] + "/deallocate?api-version=" + COMPUTE_API, {}
        ), workers))
        snapshots = list(pool.map(
            lambda item: preserve_snapshot(client, settings, preserved, item[1]), workers
        ))
    return snapshots


def preserve_snapshot(client, settings, preserved, disk_id):
    suffix = hashlib.sha256((settings["startedAtUtc"] + disk_id.lower()).encode()).hexdigest()[:16]
    snapshot_id = preserved + "/providers/Microsoft.Compute/snapshots/afhs-" + settings["date"] + "-" + suffix
    snapshot, _ = client.request("GET", snapshot_id + "?api-version=" + SNAPSHOT_API, missing_ok=True)
    if snapshot is None:
        client.operation("PUT", snapshot_id + "?api-version=" + SNAPSHOT_API, {
            "location": "japaneast", "sku": {"name": "Standard_LRS"},
            "tags": {"application": "agefreighter", "campaign": "afhs-" + settings["date"], "retain": "evidence"},
            "properties": {
                "creationData": {"createOption": "Copy", "sourceResourceId": disk_id},
                "incremental": True, "networkAccessPolicy": "DenyAll", "publicNetworkAccess": "Disabled",
            },
        })
    snapshot, _ = client.request("GET", snapshot_id + "?api-version=" + SNAPSHOT_API)
    props = snapshot["properties"]
    if props["provisioningState"] != "Succeeded" or props["creationData"]["sourceResourceId"].lower() != disk_id.lower():
        raise RuntimeError("external evidence snapshot is not verified")
    result = {"id": snapshot_id, "sourceDiskId": disk_id, "state": "Succeeded"}
    print(json.dumps({"preservedEvidenceSnapshot": result}), flush=True)
    return result


def reported_cost(client, settings):
    result, _ = client.request(
        "POST", "/subscriptions/" + settings["subscriptionId"]
        + "/providers/Microsoft.CostManagement/query?api-version=2025-03-01",
        {"type": "ActualCost", "timeframe": "MonthToDate",
         "dataset": {"granularity": "None", "aggregation": {"cost": {"name": "PreTaxCost", "function": "Sum"}}}},
    )
    properties = result["properties"]
    columns = [c["name"] for c in properties["columns"]]
    rows = properties["rows"]
    if len(rows) != 1 or rows[0][columns.index("Currency")] != "USD":
        raise RuntimeError("a single actual USD month-to-date cost is required")
    return float(rows[0][columns.index("PreTaxCost")])


def main(settings):
    preserved, disposable = validate(settings)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    print(json.dumps({"guardConfigSha256": digest, "enrolledDeletionScopes": settings["deleteTargets"],
                      "deadlineUtc": settings["deadlineUtc"]}), flush=True)
    client = Client()
    now = dt.datetime.now(UTC)
    accrued = max(0, (now - timestamp(settings["startedAtUtc"])).total_seconds()) / 3600 * settings["maxHourlyUSD"]
    reason = "deadline" if now >= timestamp(settings["deadlineUtc"]) else None
    try:
        reported = reported_cost(client, settings)
        conservative = max(reported, settings["baselineCostUSD"] + accrued) + settings["lagReserveUSD"]
        if conservative + settings["deletionReserveUSD"] + settings["maxHourlyUSD"] / 2 >= 3750:
            reason = reason or "budget"
        print(json.dumps({"reportedMtdUSD": reported, "conservativeMtdUSD": conservative,
                          "deadlineUtc": settings["deadlineUtc"], "decision": reason or "retain"}), flush=True)
    except (RuntimeError, urllib.error.URLError, KeyError, ValueError) as error:
        print(json.dumps({"costMonitoringError": str(error)}), flush=True)
        reason = reason or "budget-monitor-unavailable"
    if not reason:
        return
    preserve_worker_evidence(client, settings, preserved, disposable)
    failures = []
    for target in settings["deleteTargets"]:
        api = RESOURCE_API if target.lower() == disposable.lower() else HORIZON_API
        try:
            resource, _ = client.request("GET", target + "?api-version=" + api, missing_ok=True)
            if resource is None:
                continue
            if target.lower() != disposable.lower():
                tags = resource.get("tags", {})
                if tags.get("application") != "agefreighter" or tags.get("run") != "afht-" + settings["date"]:
                    raise RuntimeError("reused HorizonDB cluster ownership differs")
            print(json.dumps({"action": "delete", "reason": reason, "resourceId": target}), flush=True)
            client.operation("DELETE", target + "?api-version=" + api)
            remaining, _ = client.request("GET", target + "?api-version=" + api, missing_ok=True)
            if remaining is not None:
                raise RuntimeError("deletion has not been verified by a final 404")
            print(json.dumps({"deleted": target, "verifiedAbsent": True}), flush=True)
        except (RuntimeError, urllib.error.URLError, KeyError, ValueError) as error:
            failures.append({"resourceId": target, "error": str(error)})
    if failures:
        raise RuntimeError("incomplete scoped cleanup: " + json.dumps(failures))
