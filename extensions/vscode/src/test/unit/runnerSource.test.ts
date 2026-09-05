import assert from "node:assert/strict";
import test from "node:test";
import { buildSourceDraft, sourceSecrets } from "../../core/runnerSource";
import { csvFile, sourceForm, workflow } from "../sourceFixtures";

test("Neo4j 4/5 form creates TLS discovery with environment handles and no ARM lookup", () => {
  for (const location of ["azure", "on-premises", "other-cloud"] as const) {
    const draft = buildSourceDraft({ type: "neo4j", location }, sourceForm, workflow);
    const neo = (draft.configuration.source as any).neo4j;
    assert.equal(neo.uri, "neo4j+s://source.example.com:7687"); assert.equal(neo.discovery.enabled, true);
    assert.deepEqual(neo.password, { env: "AGEFREIGHTER_SOURCE_PASSWORD" });
    assert.equal(draft.canAssess, true); assert.ok(!JSON.stringify(draft).includes("resourceId"));
  }
});
test("PostgreSQL table mappings generate only quoted read queries and mapped endpoints", () => {
  const draft = buildSourceDraft({ type: "postgresql", location: "on-premises" }, { ...sourceForm, port: 5432 }, workflow);
  const pg = (draft.configuration.source as any).postgresql;
  assert.equal(pg.vertices[0].query, 'SELECT "id", "full_name", "age" FROM "public"."people" ORDER BY "id"');
  assert.equal(pg.edges[0].start.field, "from_id"); assert.equal(pg.readMode, "cursor");
  assert.deepEqual(pg.connection, { env: "AGEFREIGHTER_SOURCE_DSN" });
  assert.throws(() => buildSourceDraft({ type: "postgresql", location: "azure" }, { ...sourceForm, mappings: [{ ...sourceForm.mappings[0], collection: 'people; DROP TABLE secret' }] }, workflow), /PostgreSQL table/);
});
test("Cosmos explicit mappings bind labels and escape JSON pointers; Gremlin uses bounded discovery", () => {
  const form = { ...sourceForm, host: "account.documents.azure.com", mappings: [{ ...sourceForm.mappings[0], identity: "a/b~c" }] };
  const cosmos = (buildSourceDraft({ type: "cosmos-nosql", location: "azure" }, form, workflow).configuration.source as any).cosmos;
  assert.equal(cosmos.credential, "default-azure"); assert.equal(cosmos.vertices[0].idField, "/a~1b~0c");
  assert.deepEqual(cosmos.vertices[0].parameters, [{ name: "@label", value: "Person" }]);
  const gremlin = (buildSourceDraft({ type: "cosmos-nosql", location: "azure" }, { ...form, cosmosFormat: "gremlin" }, workflow).configuration.source as any).cosmos;
  assert.equal(gremlin.gremlin.maxDiscoveryDocuments, 10000); assert.equal(gremlin.vertices, undefined);
});
test("CSV mappings use selected identities and explicit types, but cannot run before upload", () => {
  const form = { ...sourceForm, mappings: [{ ...sourceForm.mappings[0], collection: csvFile.id, properties: "age=age:int64,active=active:boolean,tags=tags:string[]" }] };
  const draft = buildSourceDraft({ type: "csv", location: "local" }, form, workflow, [csvFile]);
  const csv = (draft.configuration.source as any).csv;
  assert.equal(draft.canAssess, false); assert.deepEqual({ ...csv.vertices[0].propertyTypes }, { age: "int64", active: "boolean", tags: "string[]" });
  assert.equal(csv.defaults.nullValue, "\\N");
  assert.equal(csv.vertices[0].path, `/var/lib/agefreighter/workflows/${workflow}/uploads/${csvFile.id}.csv`);
  assert.throws(() => buildSourceDraft({ type: "csv", location: "local" }, form, workflow, []), /picker/);
});
test("source forms reject URL credentials, invalid ports, duplicate labels, properties and unmapped endpoints", () => {
  for (const edit of [{ host: "https://user:pass@host" }, { port: 0 }, { port: 65536 }, { namespace: "a\nb" }]) {
    assert.throws(() => buildSourceDraft({ type: "neo4j", location: "on-premises" }, { ...sourceForm, ...edit }, workflow));
  }
  for (const mappings of [[sourceForm.mappings[0], sourceForm.mappings[0]], [{ ...sourceForm.mappings[0], properties: "name=a,name=b" }], [sourceForm.mappings[0], { ...sourceForm.mappings[1], startLabel: "Unknown" }]]) {
    assert.throws(() => buildSourceDraft({ type: "postgresql", location: "azure" }, { ...sourceForm, mappings }, workflow));
  }
});
test("passwords are separate from forms and PostgreSQL URI preserves special characters with strict TLS", () => {
  const password = "p@ss:/?#% secret";
  const form = { ...sourceForm, host: "2001:db8::1", port: 5432, username: "reader@tenant", database: "a/b" };
  const secrets = sourceSecrets("postgresql", form, password);
  const uri = new URL(secrets.AGEFREIGHTER_SOURCE_DSN!);
  assert.equal(decodeURIComponent(uri.password), password); assert.equal(uri.hostname, "[2001:db8::1]"); assert.equal(uri.searchParams.get("sslmode"), "verify-full");
  assert.equal(decodeURIComponent(uri.pathname.slice(1)), "a/b");
  assert.ok(!JSON.stringify(buildSourceDraft({ type: "postgresql", location: "on-premises" }, form, workflow)).includes(password));
  assert.deepEqual(sourceSecrets("cosmos-nosql", sourceForm), {});
});
