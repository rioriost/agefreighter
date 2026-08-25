# Load job configuration

agefreighter 2.x load jobs use the versioned `agefreighter.io/v2` `LoadJob`
format. YAML and JSON inputs are accepted. Documents are decoded strictly:
unknown fields, multiple YAML documents, unsupported API versions, and literal
credentials are rejected.

The canonical structural definition is
[`load-job.schema.json`](load-job.schema.json). Cross-field constraints, such as
batch memory limits and edge identity requirements for upserts, are enforced by
the CLI in addition to the schema.

Validate a job without connecting to its source or target:

```sh
agefreighter validate job.yaml
```

Print its normalized static execution plan:

```sh
agefreighter plan job.yaml
```

## Credentials

Credentials may only refer to an environment variable or file. Literal secret
values are not supported.

```yaml
connection:
  env: AGEFREIGHTER_TARGET_DSN
```

```yaml
connection:
  file: /run/secrets/age-dsn
```

## CSV options

CSV files use UTF-8 and default to comma delimiters, double-quote quoting and
escaping, a header row, and an empty null representation. Source defaults can
be overridden per vertex or edge file. Per-file options inherit every omitted
value from the source defaults.

```yaml
source:
  type: csv
  namespace: crm
  csv:
    defaults:
      delimiter: ","
      quote: '"'
      escape: '"'
      header: true
      encoding: utf-8
      nullValue: ""
    vertices:
      - label: Person
        path: people.tsv
        idColumn: person_id
        format:
          delimiter: "\t"
        properties:
          name: full_name
```

Delimiter, quote, and escape values must each contain exactly one Unicode code
point. Delimiters cannot be line breaks or equal the quote character.

## Load modes

The target modes are `create`, `replace`, `append`, and `upsert`. Every edge
mapping in an `upsert` job must provide an external edge identity field or
column. Graph names must follow the supported Apache AGE naming subset: 3–63 UTF-8
bytes, starting with a letter or underscore, ending with a letter, digit, or
underscore, and containing only letters, digits, underscores, dots, and
hyphens.
