import { isIP } from "node:net";
import { object, SourceKind, SourceSelection } from "./runner";

export interface SourceMapping {
  kind: "vertex" | "edge"; label: string; collection: string; schema: string;
  identity: string; startLabel: string; startField: string; endLabel: string; endField: string;
  properties: string;
}
export interface SourceForm {
  name: string; namespace: string; host: string; port: number; database: string; username: string;
  vertexKey: string; edgeKey: string; cosmosFormat: "explicit" | "gremlin";
  container: string; partitionKey: string; labelField: string; nullValue: string; mappings: SourceMapping[];
}
export interface SelectedCSV { id: string; name: string }
export interface SourceDraft { form: SourceForm; configuration: Record<string, unknown>; warnings: string[]; canAssess: boolean }

function text(value: unknown, label: string, maximum = 256, optional = false): string {
  if (typeof value !== "string" || value.length > maximum || /[\x00-\x1f\x7f]/.test(value) || !optional && !value.trim()) throw new Error(`Enter a valid ${label}.`);
  return value.trim();
}
function identifier(value: unknown, label: string): string {
  const result = text(value, label, 63);
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(result)) throw new Error(`${label} must use letters, digits and underscores, starting with a letter or underscore.`);
  return result;
}
function hostname(value: unknown): string {
  const host = text(value, "host", 253);
  if (!isIP(host) && (!/^[A-Za-z0-9.-]+$/.test(host) || host.split(".").some(part => !/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(part)))) throw new Error("Host must be a hostname or IP address, without a scheme, port or credentials.");
  return host;
}
function pointer(field: string): string { return "/" + field.replaceAll("~", "~0").replaceAll("/", "~1"); }
function properties(value: string, type: SourceKind): { properties: Record<string, string>; propertyTypes: Record<string, string> } {
  const result: Record<string, string> = Object.create(null), types: Record<string, string> = Object.create(null);
  for (const entry of value.split(",").map(s => s.trim()).filter(Boolean)) {
    const match = /^([^=]+)=([^:]+)(?::(.+))?$/.exec(entry);
    if (!match) throw new Error("Properties use graph_name=source_field, with optional :type for CSV.");
    const name = identifier(match[1]?.trim(), "property name"), field = text(match[2]?.trim(), "property source field");
    if (Object.hasOwn(result, name)) throw new Error("Property names must not be duplicated.");
    result[name] = type === "cosmos-nosql" ? pointer(field) : field;
    if (match[3]) {
      if (type !== "csv" || !/^(string|int64|float64|boolean)(\[\])?$/.test(match[3])) throw new Error("Explicit property types are for CSV: string, int64, float64, boolean, or their [] arrays.");
      types[name] = match[3];
    }
  }
  if (Object.keys(result).length > 128) throw new Error("Use at most 128 property mappings per label.");
  return { properties: result, propertyTypes: types };
}

/** Form values only. No YAML, SQL, arbitrary paths or credentials are accepted. */
export function buildSourceDraft(selection: SourceSelection, raw: unknown, workflow: string, files: SelectedCSV[] = []): SourceDraft {
  const value = object(raw), type = selection.type;
  if (!/^[a-f0-9-]{36}$/.test(workflow)) throw new Error("Invalid workflow identity.");
  const name = text(value.name, "migration name", 63);
  if (!/^[a-z][a-z0-9-]{2,62}$/.test(name)) throw new Error("Migration name needs 3–63 lowercase letters, digits or hyphens.");
  const form: SourceForm = { name, namespace: identifier(value.namespace, "namespace"), host: "", port: 0, database: "", username: "", vertexKey: "", edgeKey: "", cosmosFormat: "explicit", container: "", partitionKey: "", labelField: "", nullValue: "", mappings: [] };
  const warnings = ["Sampled profiles are not exact totals or proof that migration will pass. Target deployment and writes are not part of this assessment."];
  const source: Record<string, unknown> = { type, namespace: form.namespace };
  if (type === "csv") {
    if (typeof value.nullValue !== "string" || value.nullValue.length > 32 || /[\x00-\x1f\x7f]/.test(value.nullValue)) throw new Error("Enter a CSV null marker of at most 32 characters (empty is allowed).");
    form.nullValue = value.nullValue;
  }
  if (type !== "csv") form.database = text(value.database, "database");
  if (type === "neo4j" || type === "postgresql") {
    form.host = hostname(value.host); form.port = Number(value.port); form.username = text(value.username, "username");
    if (!Number.isInteger(form.port) || form.port < 1 || form.port > 65535) throw new Error("Port must be 1–65535.");
    warnings.push("TLS certificate validation is required. Use a read-only source account; an Azure VM candidate is not proof of database identity.");
  }
  if (type === "neo4j") {
    form.vertexKey = text(value.vertexKey, "vertex key property"); form.edgeKey = text(value.edgeKey, "edge key property");
    source.neo4j = { uri: `neo4j+s://${isIP(form.host) === 6 ? `[${form.host}]` : form.host}:${form.port}`, database: form.database, sourceId: workflow,
      username: form.username, password: { env: "AGEFREIGHTER_SOURCE_PASSWORD" }, fetchRows: 5000, multiLabelPolicy: "configured",
      discovery: { enabled: true, vertexKeyProperty: form.vertexKey, edgeKeyProperty: form.edgeKey } };
  } else if (type === "cosmos-nosql") {
    if (selection.location !== "azure") throw new Error("Cosmos source must use Azure placement.");
    form.host = hostname(value.host);
    if (!form.host.endsWith(".documents.azure.com")) throw new Error("Use the Azure public-cloud Cosmos NoSQL account hostname (not the Gremlin endpoint).");
    if (value.cosmosFormat !== "explicit" && value.cosmosFormat !== "gremlin") throw new Error("Select the Cosmos document format.");
    form.cosmosFormat = value.cosmosFormat;
    source.cosmos = { endpoint: `https://${form.host}:443/`, credential: "default-azure", database: form.database, pageSize: 100 };
    warnings.push("The runner managed identity needs Cosmos data-plane read permissions. No role assignment is created by this form. Reads consume RU; 10,000 sampled rows do not bound schema-discovery RU.");
    if (form.cosmosFormat === "gremlin") {
      form.container = text(value.container, "container"); form.partitionKey = text(value.partitionKey, "partition key property");
      object(source.cosmos).gremlin = { enabled: true, container: form.container, partitionKeyProperty: form.partitionKey, maxLabels: 64, maxProperties: 128, maxDiscoveryDocuments: 10000 };
    } else form.labelField = text(value.labelField, "label field");
  }
  if (type === "postgresql" || type === "csv" || type === "cosmos-nosql" && form.cosmosFormat === "explicit") {
    if (!Array.isArray(value.mappings) || value.mappings.length < 1 || value.mappings.length > 64) throw new Error("Add 1–64 vertex/edge mappings.");
    const vertices: unknown[] = [], edges: unknown[] = [], names = new Set<string>();
    for (const item of value.mappings) {
      const row = object(item);
      if (row.kind !== "vertex" && row.kind !== "edge") throw new Error("Select vertex or edge mapping.");
      const mapping: SourceMapping = { kind: row.kind, label: identifier(row.label, "graph label"), collection: text(row.collection, "table, container or selected file"), schema: type === "postgresql" ? identifier(row.schema, "PostgreSQL schema") : "", identity: text(row.identity, "stable ID field"), startLabel: "", startField: "", endLabel: "", endField: "", properties: text(row.properties ?? "", "properties", 8192, true) };
      if (names.has(mapping.label)) throw new Error("Use a unique label for each mapping, including vertex and edge labels.");
      names.add(mapping.label);
      const props = properties(mapping.properties, type);
      const built: Record<string, unknown> = { label: mapping.label, properties: props.properties };
      const field = (v: string) => type === "cosmos-nosql" ? pointer(v) : v;
      if (mapping.kind === "vertex") built[type === "csv" ? "idColumn" : "idField"] = field(mapping.identity);
      else {
        mapping.startLabel = identifier(row.startLabel, "start vertex label"); mapping.endLabel = identifier(row.endLabel, "end vertex label");
        mapping.startField = text(row.startField, "start ID field"); mapping.endField = text(row.endField, "end ID field");
        built[type === "csv" ? "externalIdColumn" : "externalIdField"] = field(mapping.identity);
        built.start = { label: mapping.startLabel, field: field(mapping.startField) }; built.end = { label: mapping.endLabel, field: field(mapping.endField) };
      }
      if (type === "postgresql") {
        const table = identifier(mapping.collection, "PostgreSQL table");
        const fields = [...new Set([mapping.identity, ...Object.values(props.properties), ...(mapping.kind === "edge" ? [mapping.startField, mapping.endField] : [])])].map(f => `"${identifier(f, "PostgreSQL column")}"`);
        built.query = `SELECT ${fields.join(", ")} FROM "${mapping.schema}"."${table}" ORDER BY "${mapping.identity}"`;
      } else if (type === "cosmos-nosql") {
        built.container = mapping.collection;
        built.query = `SELECT * FROM c WHERE c[${JSON.stringify(form.labelField)}] = @label`;
        built.parameters = [{ name: "@label", value: mapping.label }];
      } else {
        if (!files.some(f => f.id === mapping.collection)) throw new Error("Select a CSV file through the file picker first.");
        if (!/^[a-f0-9-]{36}$/.test(mapping.collection)) throw new Error("Invalid selected CSV identity.");
        built.path = `/var/lib/agefreighter/workflows/${workflow}/uploads/${mapping.collection}.csv`;
        built.propertyTypes = props.propertyTypes;
      }
      form.mappings.push(mapping); (mapping.kind === "vertex" ? vertices : edges).push(built);
    }
    const vertexLabels = new Set(form.mappings.filter(m => m.kind === "vertex").map(m => m.label));
    if (!vertexLabels.size || form.mappings.some(m => m.kind === "edge" && (!vertexLabels.has(m.startLabel) || !vertexLabels.has(m.endLabel)))) throw new Error("Every edge endpoint must reference a configured vertex label.");
    if (type === "postgresql") source.postgresql = { connection: { env: "AGEFREIGHTER_SOURCE_DSN" }, readMode: "cursor", fetchRows: 5000, vertices, edges };
    else if (type === "csv") { source.csv = { defaults: { delimiter: ",", quote: '"', escape: '"', header: true, encoding: "utf-8", nullValue: form.nullValue }, vertices, edges }; warnings.push("CSV upload is not enabled yet. These guest paths are planned only, not evidence that files exist."); }
    else Object.assign(object(source.cosmos), { vertices, edges });
  }
  return { form, warnings, canAssess: type !== "csv", configuration: {
    apiVersion: "agefreighter.io/v2", kind: "LoadJob", metadata: { name }, source,
    target: { type: "apache-age", graph: name.replaceAll("-", "_"), mode: "create", connection: { env: "AGEFREIGHTER_TARGET_DSN" }, propertyMode: "replace" },
    runtime: { memoryLimit: "4GiB", batchRows: 5000, batchBytes: "16MiB", maxSourceConcurrency: 1, maxTransformConcurrency: 1, maxTargetConnections: 8, operationTimeout: "2m" },
    errors: { malformedRecord: "fail", missingEndpoint: "error", rejectLimit: 0 }
  } };
}

/** Password is supplied by a native secret prompt, never the webview/config. */
export function sourceSecrets(type: SourceKind, form: SourceForm, password?: string): Record<string, string> {
  if (type !== "postgresql" && type !== "neo4j") return {};
  if (!password || password.length > 16000 || /[\x00\r\n]/.test(password)) throw new Error("Enter a nonempty source password without control characters.");
  if (type === "neo4j") return { AGEFREIGHTER_SOURCE_PASSWORD: password };
  const host = isIP(form.host) === 6 ? `[${form.host}]` : form.host;
  return { AGEFREIGHTER_SOURCE_DSN: `postgresql://${encodeURIComponent(form.username)}:${encodeURIComponent(password)}@${host}:${form.port}/${encodeURIComponent(form.database)}?sslmode=verify-full&connect_timeout=15` };
}
