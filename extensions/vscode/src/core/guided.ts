import * as path from "node:path";

export const guidedStateSchemaVersion = 1;

export type GuidedPhase =
  | "draft"
  | "source-connected"
  | "profiled"
  | "proposed"
  | "what-if-reviewed"
  | "deploying"
  | "deployed"
  | "target-ready"
  | "load-started"
  | "load-failed"
  | "load-committed"
  | "verifying"
  | "passed"
  | "failed"
  | "incomplete";

export interface Neo4jDraftInput {
  name: string;
  host: string;
  port: number;
  encrypted: boolean;
  database: string;
  sourceId: string;
  namespace: string;
  username: string;
  vertexKeyProperty: string;
  edgeKeyProperty: string;
}

export interface GuidedState {
  schemaVersion: 1;
  revision: number;
  id: string;
  phase: GuidedPhase;
  createdAt: string;
  updatedAt: string;
  jobPath?: string;
  source: {
    type: "neo4j";
    host: string;
    port: number;
    database: string;
    sourceId: string;
    placement: "azure" | "on-premises";
    subscriptionId?: string;
    resourceId?: string;
    declaredLocation?: string;
    resolvedLocation?: string;
    resolvedZone?: string;
    placementConfidence: "verified" | "declared" | "unknown";
  };
  profile?: {
    outcome: "pass" | "fail" | "incomplete";
    evidencePath: string;
    inventoryEvidencePath?: string;
    generatedAt?: string;
  };
  proposal?: {
    evidencePath: string;
    generatedAt: string;
    expiresAt: string;
    region: string;
    zone?: string;
    deployable: boolean;
  };
  deployment?: {
    subscriptionId: string;
    resourceGroup: string;
    deploymentName: string;
    resourceId?: string;
  };
  durableJobId?: string;
}

export interface CapacityEvidence {
  method: string;
  targetRows?: bigint;
  targetRowsLowerBound: boolean;
  recommendedStorageLow?: bigint;
  recommendedStorageHigh?: bigint;
  deployable: boolean;
  reason?: string;
}

export interface InventoryEvidence {
  vertices: bigint;
  edges: bigint;
  totalRows: bigint;
  exact: boolean;
  method: string;
}

const slugPattern = /^[a-z][a-z0-9-]{2,62}$/;
const sourceIDPattern = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export function normalizeNeo4jInput(value: Neo4jDraftInput): Neo4jDraftInput {
  const normalized: Neo4jDraftInput = {
    ...value,
    name: value.name.trim().toLowerCase(),
    host: value.host.trim(),
    database: value.database.trim(),
    sourceId: value.sourceId.trim(),
    namespace: value.namespace.trim(),
    username: value.username.trim(),
    vertexKeyProperty: value.vertexKeyProperty.trim(),
    edgeKeyProperty: value.edgeKeyProperty.trim()
  };
  const errors: string[] = [];
  if (!slugPattern.test(normalized.name)) {
    errors.push("Migration name must start with a letter and use 3-63 lowercase letters, digits, or hyphens.");
  }
  if (!isHost(normalized.host)) {
    errors.push("Host must be a hostname or address without a URL scheme, path, credentials, or control characters.");
  }
  if (!Number.isSafeInteger(normalized.port) || normalized.port < 1 || normalized.port > 65535) {
    errors.push("Port must be an integer from 1 to 65535.");
  }
  for (const [label, input, maximum] of [
    ["Database", normalized.database, 256],
    ["Namespace", normalized.namespace, 63],
    ["Username", normalized.username, 256]
  ] as const) {
    if (!isPlainValue(input, maximum)) {
      errors.push(`${label} must be 1-${maximum} characters without control characters.`);
    }
  }
  if (!sourceIDPattern.test(normalized.sourceId)) {
    errors.push("Source ID must use 1-128 letters, digits, dots, underscores, colons, or hyphens.");
  }
  for (const [label, input] of [
    ["Vertex key property", normalized.vertexKeyProperty],
    ["Edge key property", normalized.edgeKeyProperty]
  ] as const) {
    if (!isPlainValue(input, 256)) {
      errors.push(`${label} must be 1-256 characters without control characters.`);
    }
  }
  if (errors.length > 0) {
    throw new Error(errors.join(" "));
  }
  return normalized;
}

export function buildNeo4jDraftYAML(
  inputValue: Neo4jDraftInput,
  sourcePasswordPath: string,
  targetConnectionPath: string
): string {
  const input = normalizeNeo4jInput(inputValue);
  if (!path.isAbsolute(sourcePasswordPath) || !path.isAbsolute(targetConnectionPath)) {
    throw new Error("Guided secret references must use absolute paths.");
  }
  const scheme = input.encrypted ? "neo4j+s" : "neo4j";
  const uri = `${scheme}://${formatHost(input.host)}:${input.port}`;
  return [
    "apiVersion: agefreighter.io/v2",
    "kind: LoadJob",
    "metadata:",
    `  name: ${yamlString(input.name)}`,
    "source:",
    "  type: neo4j",
    `  namespace: ${yamlString(input.namespace)}`,
    "  neo4j:",
    `    uri: ${yamlString(uri)}`,
    `    database: ${yamlString(input.database)}`,
    `    sourceId: ${yamlString(input.sourceId)}`,
    `    username: ${yamlString(input.username)}`,
    "    password:",
    `      file: ${yamlString(sourcePasswordPath)}`,
    "    fetchRows: 5000",
    "    multiLabelPolicy: configured",
    "    discovery:",
    "      enabled: true",
    `      vertexKeyProperty: ${yamlString(input.vertexKeyProperty)}`,
    `      edgeKeyProperty: ${yamlString(input.edgeKeyProperty)}`,
    "target:",
    "  type: apache-age",
    `  graph: ${yamlString(input.name.replaceAll("-", "_"))}`,
    "  mode: create",
    "  connection:",
    `    file: ${yamlString(targetConnectionPath)}`,
    "  propertyMode: replace",
    "runtime:",
    "  memoryLimit: 4GiB",
    "  batchRows: 5000",
    "  batchBytes: 16MiB",
    "  maxSourceConcurrency: 1",
    "  maxTransformConcurrency: 1",
    "  maxTargetConnections: 8",
    "  operationTimeout: 2m",
    "errors:",
    "  malformedRecord: fail",
    "  missingEndpoint: error",
    "  rejectLimit: 0",
    ""
  ].join("\n");
}

export function extractCapacityEvidence(profile: unknown): CapacityEvidence {
  const document = asRecord(profile);
  const sections = Array.isArray(document?.sections) ? document.sections : [];
  const capacity = sections
    .map(asRecord)
    .find((section) => section?.title === "Capacity indicators");
  const fields = Array.isArray(capacity?.fields) ? capacity.fields.map(asRecord) : [];
  const value = (name: string): string | undefined => {
    const field = fields.find((candidate) => candidate?.name === name);
    return typeof field?.value === "string" ? field.value : undefined;
  };
  const method = value("method") ?? "unavailable";
  const rowText = value("estimatedTargetRows");
  const storage = parseRange(value("recommendedStorageBytesRange"));
  const targetRowsLowerBound = rowText?.startsWith(">=") ?? false;
  const targetRows = parseInteger(rowText?.replace(/^>=/, ""));
  const complete = method === "complete-stream-range";
  return {
    method,
    targetRows,
    targetRowsLowerBound,
    recommendedStorageLow: storage?.[0],
    recommendedStorageHigh: storage?.[1],
    deployable: complete && storage !== undefined,
    reason: complete
      ? storage === undefined ? "The profile did not provide a usable storage range." : undefined
      : "The bounded profile is a lower bound. Supply a reviewed upper bound or complete inventory before deployment."
  };
}

export function extractInventoryEvidence(inventory: unknown): InventoryEvidence {
  const document = asRecord(inventory);
  const sections = Array.isArray(document?.sections) ? document.sections : [];
  const section = sections.map(asRecord).find((candidate) => candidate?.title === "Source inventory");
  const fields = Array.isArray(section?.fields) ? section.fields.map(asRecord) : [];
  const value = (name: string): string | undefined => {
    const field = fields.find((candidate) => candidate?.name === name);
    return typeof field?.value === "string" ? field.value : undefined;
  };
  const vertices = parseInteger(value("vertices"));
  const edges = parseInteger(value("edges"));
  const totalRows = parseInteger(value("totalRows"));
  const method = value("countMethod") ?? "unavailable";
  if (vertices === undefined || edges === undefined || totalRows === undefined ||
      vertices + edges !== totalRows) {
    throw new Error("The source inventory did not provide consistent exact totals.");
  }
  return {
    vertices,
    edges,
    totalRows,
    exact: document?.outcome === "pass" && method === "neo4j-transactional-count-store",
    method
  };
}

export function combineCapacityAndInventory(
  capacity: CapacityEvidence,
  inventory: InventoryEvidence
): CapacityEvidence {
  if (!inventory.exact) {
    return { ...capacity, deployable: false, reason: "The source inventory is not exact." };
  }
  const observedRows = capacity.targetRows;
  if (observedRows === undefined || observedRows <= 0n || inventory.totalRows < observedRows ||
      capacity.recommendedStorageLow === undefined || capacity.recommendedStorageHigh === undefined) {
    return {
      ...capacity,
      targetRows: inventory.totalRows,
      targetRowsLowerBound: false,
      deployable: false,
      reason: "The bounded profile and exact inventory cannot be combined into a consistent storage estimate."
    };
  }
  return {
    method: "exact-counts-scaled-bounded-profile",
    targetRows: inventory.totalRows,
    targetRowsLowerBound: false,
    recommendedStorageLow: ceilDivide(capacity.recommendedStorageLow * inventory.totalRows, observedRows),
    recommendedStorageHigh: ceilDivide(capacity.recommendedStorageHigh * inventory.totalRows, observedRows),
    deployable: true
  };
}

export function createGuidedState(
  id: string,
  inputValue: Neo4jDraftInput,
  placement: GuidedState["source"]["placement"]
): GuidedState {
  const input = normalizeNeo4jInput(inputValue);
  const now = new Date().toISOString();
  return {
    schemaVersion: guidedStateSchemaVersion,
    revision: 1,
    id,
    phase: "draft",
    createdAt: now,
    updatedAt: now,
    source: {
      type: "neo4j",
      host: input.host,
      port: input.port,
      database: input.database,
      sourceId: input.sourceId,
      placement,
      placementConfidence: "unknown"
    }
  };
}

export function assertPersistableState(state: GuidedState): GuidedState {
  if (hasCredentialShapedKey(state)) {
    throw new Error("Guided state contains a prohibited credential-shaped field.");
  }
  return state;
}

function hasCredentialShapedKey(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some(hasCredentialShapedKey);
  }
  if (value === null || typeof value !== "object") {
    return false;
  }
  return Object.entries(value as Record<string, unknown>).some(([key, nested]) =>
    /password|credential|accessToken|refreshToken|connectionString|^dsn$/i.test(key) ||
    hasCredentialShapedKey(nested)
  );
}

function isHost(value: string): boolean {
  return value.length >= 1 && value.length <= 253 &&
    !/[\u0000-\u001f\u007f\s/@]/u.test(value) &&
    !value.includes("://") &&
    !(value.includes(":") && !/^\[[0-9a-fA-F:]+\]$/.test(value));
}

function isPlainValue(value: string, maximum: number): boolean {
  return value.length >= 1 && Buffer.byteLength(value, "utf8") <= maximum &&
    !/[\u0000-\u001f\u007f]/u.test(value);
}

function formatHost(host: string): string {
  if (host.startsWith("[") && host.endsWith("]")) {
    return host;
  }
  return host.includes(":") ? `[${host}]` : host;
}

function yamlString(value: string): string {
  return JSON.stringify(value);
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function parseRange(value: string | undefined): readonly [bigint, bigint] | undefined {
  const match = /^(\d+)\.\.(\d+)$/.exec(value ?? "");
  if (!match?.[1] || !match[2]) {
    return undefined;
  }
  return [BigInt(match[1]), BigInt(match[2])];
}

function parseInteger(value: string | undefined): bigint | undefined {
  return /^\d+$/.test(value ?? "") ? BigInt(value!) : undefined;
}

function ceilDivide(value: bigint, divisor: bigint): bigint {
  return (value + divisor - 1n) / divisor;
}
