# Cypher Compatibility Report

- Apache AGE target: `1.6`
- Files: 1
- Queries: 6

## Queries

| File | Query | Location | Classification |
|---|---:|---:|---|
| `corpus.cypher` | 1 | 1:1 | compatible-with-manual-change |
| `corpus.cypher` | 2 | 2:1 | compatible |
| `corpus.cypher` | 3 | 3:1 | unsupported |
| `corpus.cypher` | 4 | 4:1 | unknown |
| `corpus.cypher` | 5 | 5:1 | unknown |
| `corpus.cypher` | 6 | 6:1 | unknown |

## Findings

| File | Query | Location | Severity | Rule | Evidence | Remediation |
|---|---:|---:|---|---|---|---|
| `corpus.cypher` | 1 | 1:33 | warning | `AGE16-W001` | $<parameter> | bind parameters through the AGE cypher() parameter map; do not interpolate values |
| `corpus.cypher` | 2 | 2:1 | info | `AGE16-C001` | recognized bounded openCypher structure | test application behavior against the target AGE release before cutover |
| `corpus.cypher` | 3 | 3:1 | error | `AGE16-U001` | CALL <identifier> | replace the procedure call with application code or documented AGE/openCypher clauses |
| `corpus.cypher` | 3 | 3:6 | error | `AGE16-U001` | <neo4j-namespace>.<function>(…) | replace Neo4j extension calls with application code or documented AGE functions |
| `corpus.cypher` | 4 | 4:1 | unknown | `AGE16-X003` | unrecognized query entry clause | start with a documented AGE/openCypher clause |
| `corpus.cypher` | 4 | 4:1 | unknown | `AGE16-X003` | function compatibility is not in the AGE 1.6 rule catalog | verify the function against AGE 1.6 documentation or replace it with a cataloged function |
| `corpus.cypher` | 4 | 4:1 | unknown | `AGE16-X003` | statement is not fully consumed by the bounded structural grammar | correct the clause, expression, operator, or literal structure before rechecking |
| `corpus.cypher` | 5 | 5:18 | unknown | `AGE16-X003` | function compatibility is not in the AGE 1.6 rule catalog | verify the function against AGE 1.6 documentation or replace it with a cataloged function |
| `corpus.cypher` | 6 | 6:17 | unknown | `AGE16-X003` | statement is not fully consumed by the bounded structural grammar | correct the clause, expression, operator, or literal structure before rechecking |
| `corpus.cypher` | 6 | 6:26 | unknown | `AGE16-X001` | unterminated string literal | terminate the string literal before rechecking |

## Summary

- Compatible: 1
- Compatible with manual change: 1
- Unsupported: 1
- Unknown: 3
- Warnings: 1
- Compatibility score: unknown
- Conclusive: false
