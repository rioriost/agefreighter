#!/usr/bin/env python3
"""Read saved ARM JSON only; emit allowlisted evidence. Never invokes Azure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

WORKFLOW = "0d8b4bc9-102c-4d74-8270-061ce31ce163"
PREFIX = "af-0d8b4bc9102c4d748270"
BASE = "/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-af-vscode-p1-20260905-a/providers"
VM = BASE + "/Microsoft.Compute/virtualMachines/" + PREFIX
DEPLOYMENT = BASE + "/Microsoft.Resources/deployments/" + PREFIX
CONTAINER = BASE + "/Microsoft.Storage/storageAccounts/af0d8b4bc9102c4d74827006/blobServices/default/containers/af-" + WORKFLOW
ARCHIVE = "3c33a179916ec08a83ca8ccb3c19e7682862d0382a63b2c05b5369a3cb124e33"
OBSERVERS = {VM + "/runCommands/af-b09-observe-0" + str(index) for index in range(1, 4)}
BOOLEAN_FIELDS = (
    "cloudFinalFailed", "archiveHasOnlyRetainedRegularMember", "retainedMemberExtracted",
    "missingToolsExtractionLeftEmptyFile", "bootstrapCompleteAbsent", "archiveMarkerAbsent",
    "versionEvidenceAbsent", "installedExecutablesAbsent", "missingMemberDiagnosticObserved",
    "tarFailureDiagnosticObserved", "terminalPackagingFailureObserved",
)
STATES = {"Accepted", "Running", "Succeeded", "Failed", "Canceled", "TimedOut", "Creating", "Updating", "Deleting"}


def same_id(actual, expected):
    return isinstance(actual, str) and actual.isascii() and actual.casefold() == expected.casefold()


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Input must be bounded saved JSON")
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Input exceeded bound")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected an ARM JSON object")
    return value, hashlib.sha256(data).hexdigest()


def sanitized_state(value):
    return value if isinstance(value, str) and value in STATES else "Unrecognized"


def check_deployment(raw, operations=None):
    if not same_id(raw.get("id"), DEPLOYMENT):
        raise ValueError("Deployment identity differs from approved scope")
    state = sanitized_state(raw.get("properties", {}).get("provisioningState"))
    result = {"deploymentId": DEPLOYMENT, "provisioningState": state, "deploymentObservedAccepted": state in {"Accepted", "Running", "Succeeded"}}
    if operations is not None:
        rows = operations.get("value")
        if not isinstance(rows, list) or len(rows) > 50 or operations.get("nextLink"):
            raise ValueError("Operations evidence is incomplete or outside bound")
        allowed = {VM, BASE + "/Microsoft.Network/networkSecurityGroups/" + PREFIX,
                   BASE + "/Microsoft.Network/networkInterfaces/" + PREFIX,
                   CONTAINER + "/providers/Microsoft.Authorization/roleAssignments/" + WORKFLOW}
        targets = []
        for row in rows:
            properties = row.get("properties", {})
            resource = properties.get("targetResource")
            if resource:
                target = resource.get("id")
                canonical = next((item for item in allowed if same_id(target, item)), None)
                if canonical is None:
                    raise ValueError("Operation resource differs from approved scope")
                targets.append({"resourceId": canonical, "provisioningState": sanitized_state(properties.get("provisioningState"))})
        result["operationCount"] = len(rows)
        result["operationResources"] = targets
        result["operationsContainApprovedVM"] = any(item["resourceId"] == VM for item in targets)
    result["checkPassed"] = result["deploymentObservedAccepted"] and (operations is None or result["operationsContainApprovedVM"])
    return result


def check_observer(raw):
    canonical = next((item for item in OBSERVERS if same_id(raw.get("id"), item)), None)
    if canonical is None:
        raise ValueError("Observation control is outside approved VM/names")
    properties = raw.get("properties", {})
    view = properties.get("instanceView", {})
    output = view.get("output", "")
    if not isinstance(output, str) or len(output.encode()) >= 4096 or len(output.strip().splitlines()) != 1:
        raise ValueError("Expected one bounded observer JSON line")
    observation = json.loads(output)
    allowed = set(BOOLEAN_FIELDS) | {"schemaVersion", "scope", "observedAt", "bootId", "cloudInitStatus", "cloudInitExitCode", "archiveSHA256", "runnerExecutableProcessCount", "observationError"}
    if not isinstance(observation, dict) or set(observation) - allowed:
        raise ValueError("Unexpected observer evidence fields")
    result = {"commandId": canonical, "executionState": sanitized_state(view.get("executionState")), "exitCode": view.get("exitCode") if type(view.get("exitCode")) is int else None}
    boot = observation.get("bootId", "")
    result["validBootIdentity"] = isinstance(boot, str) and bool(re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", boot))
    if result["validBootIdentity"]:
        result["bootId"] = boot
    result["observerSchemaValid"] = observation.get("schemaVersion") == 1 and observation.get("scope") == "B09 dedicated negative-fixture guest observation"
    result["cloudInitTerminalError"] = observation.get("cloudInitStatus") == "error" and type(observation.get("cloudInitExitCode")) is int and observation["cloudInitExitCode"] != 0
    result["archiveMatchesApprovedFixture"] = observation.get("archiveSHA256") == ARCHIVE
    result["noRunnerExecutableProcess"] = type(observation.get("runnerExecutableProcessCount")) is int and observation["runnerExecutableProcessCount"] == 0
    result["observationErrorPresent"] = "observationError" in observation
    for field in BOOLEAN_FIELDS:
        result[field] = observation.get(field) is True
    result["checkPassed"] = (result["executionState"] == "Succeeded" and result["exitCode"] == 0
        and result["validBootIdentity"] and result["observerSchemaValid"] and result["cloudInitTerminalError"]
        and result["archiveMatchesApprovedFixture"] and result["noRunnerExecutableProcess"]
        and not result["observationErrorPresent"] and all(result[key] for key in BOOLEAN_FIELDS))
    return result


def check_readiness(raw, record):
    if record.get("id") != WORKFLOW or not same_id(record.get("vmId"), VM) or not same_id(record.get("deploymentId"), DEPLOYMENT):
        raise ValueError("Workflow identity differs from approved scope")
    command = record.get("guestCommand", {})
    command_id = command.get("id", "")
    if not isinstance(command_id, str) or not command_id.isascii() or not re.fullmatch(re.escape(VM) + r"/runCommands/af-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", command_id, flags=re.IGNORECASE) or not same_id(raw.get("id"), command_id):
        raise ValueError("Readiness command is not the retained exact control")
    view = raw.get("properties", {}).get("instanceView", {})
    error = view.get("error", "")
    if not isinstance(error, str) or len(error.encode()) > 8192:
        raise ValueError("Readiness error is outside bound")
    canonical = VM + "/runCommands/af-" + command_id.rsplit("/", 1)[1][3:].lower()
    result = {"commandId": canonical, "executionState": sanitized_state(view.get("executionState")), "exitCode": view.get("exitCode") if type(view.get("exitCode")) is int else None,
        "retainedFailedReadiness": command.get("action") == "ready" and command.get("phase") == "failed",
        "runnerARMProvisioned": record.get("phase") == "provisioned",
        "noGuestReadiness": record.get("guestReady") is None, "noReadinessReceipts": record.get("readinessReceipts") in (None, []),
        "noAssessmentOrMigration": record.get("assessment") is None and record.get("migration") is None,
        "expectedBootstrapFailureMessage": "Linux bootstrap did not complete successfully." in error}
    result["checkPassed"] = result["executionState"] == "Failed" and result["exitCode"] == 1 and all(result[key] for key in (
        "retainedFailedReadiness", "runnerARMProvisioned", "noGuestReadiness", "noReadinessReceipts", "noAssessmentOrMigration", "expectedBootstrapFailureMessage"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("deployment", "observer", "readiness"))
    parser.add_argument("input", help="Saved direct ARM GET response JSON")
    parser.add_argument("--operations", help="Optional saved direct ARM deployment operations JSON")
    parser.add_argument("--record", help="Retained production workflow JSON, required for readiness")
    args = parser.parse_args()
    try:
        raw, digest = load(args.input)
        if args.kind == "deployment":
            result = check_deployment(raw, load(args.operations)[0] if args.operations else None)
        elif args.kind == "observer":
            result = check_observer(raw)
        else:
            if not args.record:
                raise ValueError("Readiness requires the retained workflow")
            result = check_readiness(raw, load(args.record)[0])
        result.update(schemaVersion=1, kind=args.kind, savedResponseSHA256=digest, localCheckOnly=True, realServiceProvenanceMustBeVerifiedSeparately=True)
    except Exception:
        result = {"schemaVersion": 1, "kind": args.kind, "checkPassed": False, "localCheckOnly": True, "error": "Saved evidence failed bounded scope/schema checks; inspect private input without publishing raw output"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["checkPassed"] else 2


if __name__ == "__main__":
    sys.exit(main())
