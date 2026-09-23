# B07 active-inventory bracketing observer — preparation only

No Azure call, guest observation, source query, signal or worker change was
performed during this preparation. **Nine offline tests pass.** These tests
forbid subprocess/network execution and use fabricated data only.

| Artifact | SHA-256 |
|---|---|
| [`scripts/b07-observe-inventory.py`](scripts/b07-observe-inventory.py) | `4e49b3fb10a6566b648a1ccfb0c519cc78d34a99e2db3fa1f196073264429464` |
| [`scripts/prepare-b07-observer.py`](scripts/prepare-b07-observer.py) | `7a95d7bd018a0ae4e8cdacc76dc3b4aa317f8f8bd81be310a95d34aff8164a27` |
| Unchanged [`scripts/b10-observe-inventory.py`](scripts/b10-observe-inventory.py) dependency | `0d35975ef67274123a481a57a37fccf6f5df8683ad02bf1ea62fa7ca5b170e37` |

Independent review found no blocking issues in the final observer/builder,
including its nine offline tests and distinct approval/first-VM-intent clocks.
The four-hour authorization window and the separate ninety-minute compute
window are both enforced; approval time is never relabeled as the later
compute start.

## Exact operation binding and execution budget

Only workflow `0f83d520-cd03-4792-8e38-3f146abfde39` and VM
`af-0f83d520cd0347928e38` in the already specified P1 group/subscription are
accepted. The full VM ARM ID is fixed in the observer. A metadata-only IMDS
`compute/resourceId` GET verifies the actual executing VM before reading worker
state. It requests no token or credentials, refuses redirects/proxies, and has
a five-second/2048-byte bound. This uses Microsoft's documented
[Azure instance metadata service](https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service).

The new second-inventory operation UUID, boot UUID and **guest** configuration
SHA-256 must come from that actual operation's ordinary production response,
not the first inventory, an edited record or a guessed desktop-only hash.
The private local template is
`production-simulation/work/b02-b06-b07-20260924/b07-observer/binding.template.json`.
Its null fields deliberately prevent execution. Fill only after actual approval,
first VM intent and the second inventory exist. `notAfterUTC` must be no later
than both first VM intent + 90 minutes and actual approval + 4 hours.

Use the local-only builder with exact binding-file, observer and base hashes:

```
python3 scripts/prepare-b07-observer.py \
  --binding /absolute/private/path/binding.json \
  --binding-sha256 ACTUAL_BINDING_FILE_SHA256 \
  --observer-sha256 4e49b3fb10a6566b648a1ccfb0c519cc78d34a99e2db3fa1f196073264429464 \
  --base-sha256 0d35975ef67274123a481a57a37fccf6f5df8683ad02bf1ea62fa7ca5b170e37 \
  --output /absolute/private/path/new-exclusive-output-directory
```

This creates exactly `before-body.json` and `after-body.json`, plus private
hash/identity metadata. It does not launch either body. Both bodies embed
hash-checked reviewed Python in memory, and have `timeoutInSeconds: 60` and
`asyncExecution: false`. Their resource IDs are deterministic children of the
exact new VM: `af-b07-before-<operation>` and `af-b07-after-<operation>`.
The scope permits at most **two new Run Command observations**, counted inside
the overall sixty-definition lifetime limit. The local builder is not a
cloud dispatcher or an independent authorization/counting mechanism.

Before either separately approved submission, require exact ownership, available
command capacity, an exact-ID 404 and a durable create-only intent in the root's
submission journal. Submit each body at most once. A lost/failed reply is
GET-only reconciliation, never a second PUT. No third observation is approved.

## Observed fields and refusal behavior

Each call performs two exact-unit `systemctl show` diagnostics and bounded reads
of the unit cgroup, numeric process identity and the exact operation's
`state.json`. It requires actual inventory/running state, exact workflow,
operation, boot and guest configuration hash before and after observation.
It checks the expected tools main process and exactly one direct inventory
child, executable links, unit cgroup membership, PID/start-tick identity,
InvocationID, running state and disabled restarts. Go children created from
non-main threads are found through the unit's bounded cgroup enumeration.

The existing B10 process/diagnostic functions are reused; its old workflow
constant, main function, journal and health collection are not invoked. Only
numeric process metadata, fixed executable names and allowlisted state identity
are emitted. No `job.json`, `secrets.json`, environment, command line, source
values or raw diagnostic output is read or emitted. No worker is signaled;
the reused bounded diagnostic may terminate only its own newly spawned
`systemctl` child on timeout. No guest file is written by this observer.

Internal checks stop accepting observations after 45 seconds, while the outer
managed command has the sixty-second execution bound. Output is at most 4096
bytes. Any wrong VM, missing/finished worker, changed identity, unreadable state,
timeout or malformed observation returns a fixed **inconclusive** result with
no retry authorization. Inconclusive never means finished or successful.

## Root's independent GUI bracketing requirement

1. Submit `before` and retain its successful bounded receipt and actual control
   ID. It must finish before the actual normal GUI resize-refusal interaction.
2. Observe and retain the expected GUI refusal for the same workflow while its
   second inventory remains active. Do not pause, stop, delay or restart a
   worker to obtain the observation.
3. Submit `after` only after the GUI refusal. Retain its successful receipt.
4. Compare exact binding SHA, VM/workflow/operation/boot, InvocationID, main and
   child PIDs and start ticks across both receipts. Both must show running
   inventory. Retain local request/response/UI ordering as well as guest times;
   do not silently equate clocks on different machines.

Two passing receipts alone are insufficient without the actual GUI refusal and
its causal position between them. If the worker finishes before the second
observation, preserve evidence and leave active-worker acceptance unqualified.
This is an active inventory/source-worker proof, not an active migration/load,
continuous health monitor, complete inventory result or source-data validation.
