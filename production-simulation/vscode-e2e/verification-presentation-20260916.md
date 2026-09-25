# Verification presentation negative-path qualification

Scope: B12, synthetic reports in isolated real VS Code Extension Hosts.
This is not an Azure fault trial, signed-in GUI qualification, or a replacement
for the accepted base-route and CSV recovery P1 evidence.

## Finding and correction

The production report panel used the unconditional tab title **Verified
AGEFreighter migration**, including when the controller's decision was `fail`
or `incomplete`. The verification logic did not return PASS for those cases,
but the tab label could misleadingly suggest success.

The production controller now delegates rendering to one shared panel function.
Titles and headings distinguish counts-verified, failed and incomplete states.
A counts PASS explicitly says full property-digest qualification is separate.
The raw report is escaped, placed under a retained-evidence disclosure, and
cannot select the verdict. Summary text is escaped; scripts and local resource
access remain disabled. An unexpected runtime decision defaults to incomplete.
No CLI behavior, migration configuration, graph or acceptance gate changed.

## Cases

The real host tests run the production count assessor and panel function with:

- Matching complete evidence (counts-only PASS control).
- Wrong job identity.
- Evidence generated before the job.
- Missing counts.
- Incomplete coverage despite the document claiming PASS.
- Source/target count mismatch.
- Rejected records.
- Failed integrity check.
- Truncated JSON.
- Wrong independently supplied report hash.

Assertions cover the decision, actual panel title/tab registration, HTML
heading/scope, disabled scripts and empty local-resource roots. These are
Extension Host and HTML assertions, **not** pixel-level rendering or manual
click-through of every native import dialog. Digest/controller-import failure
interaction and the matching installed signed-in candidate remain open in B12.

## Evidence

- TypeScript typecheck: PASS.
- Unit tests: **211/211 PASS** (four new display/security regressions).
- VS Code **1.136.1**: **13/13 Extension Host tests PASS**, exit 0.
- VS Code **1.105.0** (minimum engine): **13/13 PASS**, exit 0.
- Current installed VS Code executable **1.137.0**, separate disposable profile:
  **13/13 PASS**, exit 0. The first launch attempt used the obsolete `Electron`
  executable name and failed before starting a host; inspecting `Info.plist`
  identified `Code`, and the corrected run passed.
- Ten report cases plus three pre-existing command/source/wizard cases make up
  each 13-case host suite.
- Each run creates a fresh temporary user-data and extensions directory.
  Workspace trust is disabled only in the disposable test profile; the user's
  signed-in profile, credentials and retained job records are not modified.
- No new VSIX was installed into the user's profile by these tests; the tests
  load the development extension in the separate host.
- The newer hosts logged a blocked webview request during rapid panel turnover;
  all panel/tab assertions passed. This is another reason not to describe this
  batch as screenshot/rendering qualification. No security setting was weakened
  to suppress the diagnostic.

Read-only Azure inventory at approximately 04:47 UTC confirms all six retained
VMs deallocated and all fifteen Flexible Servers stopped. This batch did not
start or mutate Azure resources. The USD 800 ceiling and September 20 deadline
are unchanged; storage/Cosmos charges continue.

## Review

- Shared rendering avoids testing a parallel test-only display path.
- No test-only command is registered in the shipped extension.
- Synthetic reports never enter an accepted workflow's report store.
- Counts success is explicitly distinguished from canonical qualification.
- The new tests add actual host/tab evidence but do not close the remaining
  signed-in GUI, digest failure, network-source recovery or Gremlin trials.
