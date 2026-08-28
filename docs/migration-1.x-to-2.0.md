# Migrating from agefreighter 1.x to 2.0

agefreighter 2.0 is a Go rewrite with a new command-line interface, load-job
schema, configuration defaults, and runtime model. It does not load or convert
1.x configuration files, and installing the 2.x `agefreighter` executable can
replace the 1.x command on `PATH`. Treat the upgrade as a new deployment rather
than an in-place package update.

The 2.x release line is maintained on `main`. The Python 1.x maintenance line
and its source remain on [`release/1.x`](https://github.com/rioriost/agefreighter/tree/release/1.x).
Keep the working 1.x environment and its configuration until the 2.x migration
has been verified.

## Safe migration sequence

1. Record the installed 1.x version, source connection settings, target graph,
   labels, identity fields, edge endpoints, and any generated or transformed
   properties. Back up the target database according to the PostgreSQL
   operator's procedure.
2. Download the 2.x archive into a separate directory and verify its checksum
   and GitHub build-provenance attestation. Invoke it by absolute path until
   cutover so the 1.x command remains available.
3. Create a v2 `LoadJob` from the closest validated example in
   `internal/config/testdata/valid`. Credentials must be referenced through
   environment variables or files; literal secrets are rejected.
4. Validate the file and inspect the execution plan without connecting to the
   source or target:

   ```sh
   /path/to/v2/agefreighter validate job.yaml
   /path/to/v2/agefreighter plan job.yaml
   ```

5. Use a separate target graph for the first run. For a bounded proof of
   concept, add a `trial` block to a `create` or `replace` job. Trial jobs are
   intentionally non-resumable.
6. Run the migration, then verify the committed graph:

   ```sh
   /path/to/v2/agefreighter load job.yaml
   /path/to/v2/agefreighter status --target job.yaml JOB_ID
   /path/to/v2/agefreighter verify --target job.yaml JOB_ID
   ```

7. Compare vertex and edge counts, labels, stable identities, endpoint
   mappings, and representative property values with the 1.x result. Exercise
   application queries against the verified graph before cutover.
8. Put the verified 2.x binary on `PATH`. Retain the 1.x environment and the
   previous graph or database backup through the rollback window. For a
   committed `replace`, run `cleanup` only after that window closes.

## Configuration and command changes

| 1.x concept | 2.x replacement |
|---|---|
| Python package and dynamically installed source modules | Self-contained `agefreighter` and `agefreighter-tools` binaries |
| Interactive or command-specific connection arguments | Declarative `apiVersion: agefreighter.io/v2`, `kind: LoadJob` file |
| Literal credentials in command/config input | Environment-variable or file references |
| `load` without a separate validation stage | `validate`, `plan`, then `load` |
| `convert` | `agefreighter-tools convert-gremlin` |
| `view` | No direct 2.x equivalent; use an AGE-compatible Cypher client |
| Generated/diagnostic helper commands | `agefreighter-tools`; see the [tools reference](reference/tools.md) |

There is no automatic 1.x configuration converter. This is deliberate: v2
requires explicit stable identities, edge endpoint mappings, error policy, and
target behavior so that resume and verification are deterministic. See the
[configuration reference](reference/configuration.md) and
[operations guide](reference/operations.md) while translating a job.

## Rollback

Before cutover, rollback is simply continued use of the untouched 1.x
environment and target. After cutover, stop new writes, restore the database
backup or redirect the application to the retained graph, and re-enable the 1.x
process. Do not run `cleanup` on a retained `replace` backup until rollback is
no longer required.

Report migration defects through a GitHub issue. Report suspected security
vulnerabilities privately as described in [`SECURITY.md`](../SECURITY.md).
