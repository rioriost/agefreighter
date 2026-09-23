#!/usr/bin/env python3
"""Read-only audit of the frozen September 23 B10 receipts and private store.

No Azure, subprocess, credential, or UI access. Output contains check names only;
private records and exception details must never be printed. A passing audit
confirms retained evidence integrity, not a new cloud/GUI qualification.
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(value.replace(" UTC", "+00:00").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone required")
    return parsed


def audit(ledger, receipts, record, reports):
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})

    evidence = {}
    for name, expected in ledger["sourceReceipts"].items():
        raw = receipts[name]
        check("receipt hash: " + name, sha(raw) == expected)
        evidence[name] = json.loads(raw)
    workflow, operation = ledger["workflow"], ledger["operation"]
    configuration = ledger["configurationSHA256"]
    identity = ledger["guestIdentityAcrossCrash"]
    submission = evidence["inventory-submission.json"]
    final = evidence["final-import.json"]
    status = evidence["reconnected-status.json"]
    crash = evidence["host-crash.json"]
    prior = ledger["priorHistoryAtSubmission"]
    imported = ledger["reconciliationAndImport"]
    for name, value, action in [("submission", submission, "inventory"),
                                ("reconnected status", status, "status"),
                                ("final export", final, "export-report")]:
        check(name + " operation/configuration binding",
              value["assessment"]["operation"] == operation
              and value["assessment"]["configurationSHA256"] == configuration
              and value["assessment"]["bootId"] == identity["bootId"]
              and value["guestCommand"]["operation"] == operation
              and value["guestCommand"]["action"] == action)
    check("submission time", submission["guestCommand"]["submittedAt"] == ledger["submittedAt"])
    observations = []
    for label in ["before", "after"]:
        receipt = evidence["observer-" + label + ".json"]
        value = json.loads(receipt["output"])
        observations.append(value)
        check(label + " observer success", receipt["exitCode"] == 0
              and len(receipt["output"].encode()) == identity["bothObservations"]["observerOutputUTF8Bytes"])
        check(label + " worker proof", value["workflow"] == workflow
              and value["operation"] == operation and value["unit"] == identity["unit"]
              and value["bootId"] == identity["bootId"] and value["readOnly"] is True
              and value["activeInventoryProcessProven"] is True
              and value["cgroupProcessEnumerationComplete"] is True
              and value["cgroupProcessCount"] == 2)
        for stage in ["Before", "After"]:
            service = value["service" + stage]
            check(label + " " + stage + " service", service["activeState"] == "active"
                  and service["subState"] == "running" and service["restartDisabled"] is True
                  and service["NRestarts"] == 0 and service["invocationId"] == identity["invocationId"])
            main = value["main" + stage]
            check(label + " " + stage + " main identity",
                  all(main[k] == v for k, v in identity["main"].items())
                  and main["exactUnitCgroupMember"] is True)
        children = value["inventoryChildrenBefore"] + [value["inventoryChildAfter"]]
        check(label + " child identity", len(children) == 2 and all(
            all(child[k] == v for k, v in identity["directInventoryChild"].items())
            and child["exactUnitCgroupMember"] is True for child in children))
        check(label + " observed health", value["healthWithinObservedBounds"] is True
              and value["diskUsedPercent"] < 80 and value["swapUsedBytes"] == 0
              and value["kernelOOMMatchingLineCount"] == 0
              and all(value["cgroupMemory"][k] == 0 for k in ["memory.swap.current", "oom", "oom_kill"]))
    host = ledger["localHostExecutorReceipt"]
    check("executor host identity/exit", crash["host"] == host["hostPID"]
          and crash["parent"] == host["parentPID"] and crash["app"] == host["application"]
          and crash["signal"] == "SIGKILL" and crash["mainLogExitCode"] == 9
          and crash["autoRestartedHost"] == host["automaticReplacementHostPID"]
          and crash["workflowLockObservedAfterExit"] is False)
    # VS Code's raw local log uses no suffix; the receipt's startLocal binds JST.
    exit_time = datetime.fromisoformat(crash["mainLogExitLocal"]).replace(
        tzinfo=utc(crash["startLocal"]).tzinfo)
    check("retained timestamp ordering", utc(observations[0]["finishedUTC"]) < exit_time
          < utc(observations[1]["startedUTC"]) < utc(imported["reportGeneratedAt"]))
    check("host exit time binding", exit_time == utc(ledger["timeline"]["hostExitUTCFromExecutorMainLogReceipt"]))
    check("history retained", submission["history"] == final["history"] == record["assessmentHistory"]
          and len(final["history"]) == 1 and final["history"][0]["operation"] == prior["operation"])
    check("normal store operation/assessment", record["id"] == workflow
          and record["assessment"] == final["assessment"] and record["assessment"]["phase"] == "finished")
    check("source draft unchanged", sha(json.dumps(record["sourceDraft"], separators=(",", ":"),
          ensure_ascii=False).encode()) == final["sourceDraftSHA"] == imported["sourceDraftSHA256"])
    check("no target or migration", not record.get("target") and not record.get("migration")
          and final["targetPresent"] is False and final["migrationPresent"] is False)
    for name, expected in [(operation, imported), (prior["operation"], prior)]:
        raw = reports[name]
        check("retained report seal: " + ("current" if name == operation else "prior"),
              len(raw) == expected["reportBytes"] and sha(raw) == expected["reportSHA256"])
    report = json.loads(reports[operation])
    check("report content matches receipt", report == final["report"])
    check("complete successful source report", report["outcome"] == "pass"
          and not report["errors"] and not report["incompleteChecks"]
          and {v["id"]: v["status"] for v in report["checks"]} == {"read-only": "pass", "source-counts": "pass"})
    counts = next(s["fields"] for s in report["sections"] if s["title"] == "Mapped record counts")
    check("all 18 labels and 5.6M records", len(counts) == 18 and all(v["status"] == "pass" for v in counts)
          and sum(int(v["value"]) for v in counts if v["name"].startswith("vertex:")) == 1600000
          and sum(int(v["value"]) for v in counts if v["name"].startswith("edge:")) == 4000000)
    check("sealed imported transfer", any(t["operation"] == operation and t["phase"] == "imported"
          and t["bytes"] == imported["reportBytes"] and t["sha256"] == imported["reportSHA256"]
          for t in record["reportTransfers"]))
    controls = evidence["final-controls.json"]
    names = {c["name"] for c in controls}
    bounded = ledger["boundedNoReplayEvidence"]
    check("bounded final controls", len(controls) == len(names) == bounded["finalManagedControlCount"] == 19
          and all(c["state"] == "Succeeded" for c in controls)
          and set(bounded["fiveNewControlNamesAndActions"]).issubset(names))
    stop = evidence["verified-stop.json"]
    check("retained stopped-state evidence", stop["runner"] == "af-ae9523105eba42b69fe3"
          and stop["source"] == "afpg-p1-source-20260907"
          and "PowerState/deallocated" in stop["runnerStatuses"]
          and stop["sourceState"] == {"public": "Disabled", "state": "Stopped"}
          and stop["sourceDiskDataEvidencePreserved"] is True
          and utc(stop["verifiedUTC"]) == utc(ledger["shutdown"]["verifiedUTC"])
          and utc(stop["verifiedUTC"]) < utc(stop["hardStopUTC"]))
    return {"schemaVersion": 1, "auditKind": "offline-retained-B10-evidence-integrity",
            "pass": all(c["pass"] for c in checks), "checks": checks,
            "limits": ["No fresh cloud, GUI, process, or credential observation.",
                       "GUI, raw host binding, baseline control count and shutdown remain executor-attributed.",
                       "Timestamp ordering does not independently measure cross-machine clock offset.",
                       "Control receipts support bounded no-replay evidence, not instrumented HTTP counts."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--check-rejections", action="store_true",
                        help="Check in-memory corrupted copies; never alter retained evidence")
    args = parser.parse_args()
    try:
        ledger = json.loads(args.ledger.read_bytes())
        receipts = {n: (args.receipts / n).read_bytes() for n in ledger["sourceReceipts"]}
        record = json.loads((args.store / (ledger["workflow"] + ".json")).read_bytes())
        reports = {op: (args.store / (ledger["workflow"] + ".report-" + op + ".json")).read_bytes()
                   for op in [ledger["operation"], ledger["priorHistoryAtSubmission"]["operation"]]}
        result = audit(ledger, receipts, record, reports)
        if args.check_rejections and result["pass"]:
            negatives = []

            def rejected(name, changed_ledger=ledger, changed_receipts=receipts,
                         changed_record=record, changed_reports=reports, required_failure=None):
                outcome = audit(changed_ledger, changed_receipts, changed_record, changed_reports)
                failures = [c["check"] for c in outcome["checks"] if not c["pass"]]
                negatives.append({"check": name, "pass": not outcome["pass"]
                                  and (required_failure is None or required_failure in failures)})

            tampered = dict(receipts)
            tampered["inventory-submission.json"] += b"\n"
            rejected("changed receipt bytes", changed_receipts=tampered)
            changed_reports = dict(reports)
            changed_reports[ledger["operation"]] += b"\n"
            rejected("changed current report bytes", changed_reports=changed_reports)
            changed_reports = dict(reports)
            changed_reports[ledger["priorHistoryAtSubmission"]["operation"]] += b"\n"
            rejected("changed prior report bytes", changed_reports=changed_reports)
            changed_record = copy.deepcopy(record)
            changed_record["assessmentHistory"] = []
            rejected("removed prior history", changed_record=changed_record)
            changed_record = copy.deepcopy(record)
            changed_record["reportTransfers"] = []
            rejected("missing imported transfer", changed_record=changed_record)
            changed_receipts = dict(receipts)
            observation = json.loads(changed_receipts["observer-after.json"])
            # Equal-length substitution keeps the byte-length gate intact.
            observation["output"] = observation["output"].replace('"168071"', '"999999"')
            changed_receipts["observer-after.json"] = json.dumps(observation).encode()
            changed_ledger = copy.deepcopy(ledger)
            changed_ledger["sourceReceipts"]["observer-after.json"] = sha(changed_receipts["observer-after.json"])
            rejected("different guest child despite resealed receipt", changed_ledger,
                     changed_receipts=changed_receipts, required_failure="after child identity")
            result["inMemoryRejectionChecks"] = negatives
            result["pass"] = all(v["pass"] for v in negatives)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        result = {"schemaVersion": 1, "pass": False, "error": "Required evidence missing, malformed, or unreadable"}
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
