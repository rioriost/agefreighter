#!/usr/bin/env python3
import json
import time
import urllib.parse
import urllib.request

http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
metadata_headers = {"Metadata": "true"}


def read_json(url, headers):
    with http.open(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


compute = read_json(
    "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01",
    metadata_headers,
)
tags = {tag["name"]: tag["value"] for tag in compute["tagsList"]}
if (
    tags.get("application") != "agefreighter"
    or tags.get("phase") != "tiny"
    or tags.get("purpose") != "horizondb-qualification"
    or not compute["name"].startswith("afht-")
    or not compute["name"].endswith("-runner")
    or tags.get("run") != compute["name"].removesuffix("-runner")
):
    raise SystemExit("Refusing to stop pools outside the dedicated tiny environment")

token = read_json(
    "http://169.254.169.254/metadata/identity/oauth2/token?"
    "api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F",
    metadata_headers,
)["access_token"]
headers = {"Authorization": "Bearer " + token, "Content-Length": "0"}
prefix = tags["run"]
operations = []
for version in ("17", "18"):
    base = (
        "https://management.azure.com/subscriptions/"
        + compute["subscriptionId"]
        + "/resourceGroups/"
        + compute["resourceGroupName"]
        + "/providers/Microsoft.HorizonDb/clusters/"
        + prefix
        + "-pg"
        + version
        + "/pools/writer"
    )
    request = urllib.request.Request(
        base + "/stop?api-version=2026-10-01-preview",
        headers=headers,
        method="POST",
    )
    with http.open(request, timeout=60) as response:
        operation = response.headers.get("Azure-AsyncOperation") or response.headers.get("Location")
        if response.status == 202 and not operation:
            raise RuntimeError("Pool stop accepted without a polling URL")
        if operation:
            parsed = urllib.parse.urlsplit(operation)
            if parsed.scheme != "https" or parsed.netloc != "management.azure.com":
                raise RuntimeError("Unexpected pool operation origin")
            operations.append(operation)

for operation in operations:
    for _ in range(120):
        state = read_json(operation, {"Authorization": headers["Authorization"]})["status"]
        if state == "Succeeded":
            break
        if state in ("Failed", "Canceled", "Cancelled"):
            raise RuntimeError("Tiny pool stop failed: " + state)
        time.sleep(10)
    else:
        raise TimeoutError("Tiny pool stop did not complete")
print("Both dedicated tiny pools stopped; data and evidence retained.")
