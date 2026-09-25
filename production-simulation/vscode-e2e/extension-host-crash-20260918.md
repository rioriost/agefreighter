# Isolated Extension Host forced-exit qualification — September 18

## Scope and verdict

**PASS for two local readiness-control crash boundaries**, using actual VS Code
1.138.0 arm64 Extension Host processes, the production receipt-removal controller
and `RunnerStore`. Azure observations and deletion are inert fixtures. This is
not a signed-in Azure trial, live control deletion, or a crash during an active
cloud inventory/migration/verifier. B09/B10 therefore remain partial.

The user's continuation was executed without starting compute, modifying Azure,
accessing credentials, replacing the installed candidate, or changing accepted
workflow evidence. Each test uses a new private profile/extensions directory.
The test harness is excluded from VSIX packaging and is not registered as an
operator command. It kills only its own development Extension Host after checking
the expected development-extension path and dedicated test-root configuration.

## Reproducible test

From `extensions/vscode`, on this Mac:

```sh
VSCODE_TEST_EXECUTABLE='/Applications/Visual Studio Code.app/Contents/MacOS/Code' npm run test:host:crash
```

Each scenario has two separate VS Code launches:

1. Use synthetic successful/deallocated ARM observations to build a real
   archive and durable removal intent under a real workflow lock. Send SIGKILL
   to the current Extension Host at the selected boundary. VS Code's independent
   main log must record the exact PID's forced exit (code 9 on this host), and
   the crash run must fail, not return successful test completion.
2. Reopen the same isolated store in a different Extension Host PID. Verify
   unchanged intent and archive hashes, retained `submitted` phase, and that
   `RunnerStore.exclusive` refuses the crash lock before any transport request.
   **Do not remove the lock.** Separately call the recovery controller with
   inert persistence/ARM adapters: present control stays uncertain; fixture 404
   yields an in-memory absent result. Neither branch dispatches DELETE or changes
   the crash-locked record. This does not claim successful native recovery past
   an unresolved crash lock.

## Observed evidence

Final run: **2026-09-18 06:23:37–06:23:49 UTC**; retained private evidence and
VS Code logs: `/tmp/af-host-crash-Es4h7O`.

| Boundary | Killed PID / reopened PID | Synthetic dispatch count before crash | Recovery controller requests | Result |
|---|---|---|---|---|
| Durable intent saved, before dispatch | 59144 / 59272 | 0 | One fixture GET; no persist | PASS |
| Synthetic dispatch accepted, before reply | 59404 / 59520 | 1 | One fixture GET; inert persist only | PASS |

All four launches have expected exit codes: **9, 0, 9, 0**. Both crash locks
remain retained for inspection. Real cloud requests from the tested controller:
**zero**. This does not assert that the VS Code application itself performs no
background network activity.

Before-dispatch archive SHA-256:
`26862f72c42da035e1f08a85667bc598b24c2019c90cc5db3c7263d1ddf8956e`.
Record SHA-256 before/after:
`556daacc2f2ead9d46cc210752e77f05bd76b0bb550e9be3c411247b26df2000`.

After-dispatch archive SHA-256:
`10295c7f9a5be96e6fb0d4657a21e9f6797b147380aba6249fb03d56d7db72da`.
Record SHA-256 before/after:
`02c7a8c1b7f2f6007278818bd86ee42a1b59ebffd48a7760835b219dde6c6b7a`.

The signed-in operator store still has **70 unchanged files / 19 workflows**;
aggregate filename/content-hash SHA-256 remains
`fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
Installed and rebuilt extension bundle hashes remain identical:
`965a4c4422ff2816668611d937123f9f05377b59bb2cd848f72a29222e74f6a3`.
Production extension behavior is unchanged; this batch adds test infrastructure.

## Validation, harness corrections and limits

- TypeScript/build and **380/380 unit tests** pass; existing isolated native
  Extension Host suite **13/13** passes. Final crash runner compiles and both
  scenarios pass with independent main-log assertions.
- Initial harness attempts failed before injection: macOS IPC rejected an
  overly long temporary profile path, and the process-argument guard did not
  match this VS Code version. Short private `/tmp` paths and the actual
  development-extension identity check correct those test assumptions. A
  proposed `Extension.extensionMode` check was rejected by typechecking and
  replaced with the supported extension path check. These were harness failures,
  not AGEFreighter migration failures; initial diagnostic directories remain.
- An earlier complete run remains at `/tmp/af-host-crash-ZACrod`; the final run
  additionally requires independent VS Code main-log forced-exit evidence.
- No automatic stale-lock clearing, retry policy, native confirmation bypass,
  legacy receipt adoption or cloud cleanup was added. These test fixtures are
  never written to the signed-in operator's store.
- Windows/Linux host behavior, power-loss durability, externally changed ARM
  resources, native operator-approved crash-lock recovery, and a signed-in
  crash with an active Azure operation remain unqualified.

Next live acceptance still needs an eligible sealed readiness receipt, fresh
resource/budget/time gates and exact action-time approval. Historical legacy
commands must not be adopted or altered to manufacture that evidence.
