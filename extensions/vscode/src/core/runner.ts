import { createHash } from "node:crypto";

export type SourceKind = "neo4j" | "postgresql" | "cosmos-nosql" | "csv";
export type SourceLocation = "azure" | "on-premises" | "other-cloud" | "local";
export const discoverySizes = ["Standard_B2s_v2", "Standard_D2s_v5", "Standard_D4s_v5"] as const;
export interface SourceSelection {
  type: SourceKind;
  location: SourceLocation;
  resourceId?: string;
}
export interface RunnerInput {
  subscriptionId: string;
  /** Shared deployment group for this runner and the later Flexible Server target. */
  resourceGroup: string;
  region: string;
  zone: string;
  subnetId: string;
  size: string;
  source: SourceSelection;
}
export interface RunnerArtifact { version: string; url: string; sha256: string }
export interface RunnerRecord {
  schemaVersion: 2;
  id: string;
  phase: "previewed" | "deployment-submitted" | "provisioned" | "failed" | "unknown";
  input: RunnerInput;
  artifact: RunnerArtifact;
  deploymentId: string;
  vmId: string;
  template: Record<string, unknown>;
  previewHash: string;
  expiresAt: string;
  updatedAt: string;
  hourlyComputeUSD: number;
}

export function sourceLocations(type: SourceKind): SourceLocation[] {
  return type === "csv" ? ["local"] : type === "cosmos-nosql" ? ["azure"] : ["azure", "on-premises", "other-cloud"];
}

export function parseRunnerInput(value: unknown): RunnerInput {
  const v = object(value);
  const source = object(v.source);
  if (!["neo4j", "postgresql", "cosmos-nosql", "csv"].includes(String(source.type))) throw new Error("Select a supported source type.");
  const type = source.type as SourceKind;
  if (!sourceLocations(type).includes(source.location as SourceLocation)) throw new Error("The source location is incompatible with its type.");
  const input: RunnerInput = {
    subscriptionId: field(v, "subscriptionId", /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i),
    resourceGroup: field(v, "resourceGroup", /^[\w().-]{1,90}$/u),
    region: field(v, "region", /^[a-z0-9]{1,64}$/),
    zone: field(v, "zone", /^[123]$/),
    subnetId: field(v, "subnetId", /^\/subscriptions\/[a-f0-9-]{36}\/resourceGroups\/[^/]+\/providers\/Microsoft.Network\/virtualNetworks\/[^/]+\/subnets\/[^/]+$/i),
    size: field(v, "size", /^Standard_[A-Za-z0-9_]+$/),
    source: { type, location: source.location as SourceLocation }
  };
  if (input.resourceGroup.endsWith(".")) throw new Error("Resource group names cannot end in a period.");
  if (!discoverySizes.some(size => size === input.size)) throw new Error("Select a supported Linux x64 discovery SKU.");
  if (input.subnetId.split("/")[2]?.toLowerCase() !== input.subscriptionId.toLowerCase()) throw new Error("The runner subnet must belong to the runner subscription.");
  if (source.resourceId !== undefined && source.resourceId !== "") {
    input.source.resourceId = field(source, "resourceId", /^\/subscriptions\/[a-f0-9-]{36}\/resourceGroups\/[^/]+\/providers\/[^/]+\/.+$/i);
  }
  if (input.source.location === "azure" && !input.source.resourceId) throw new Error("Select or enter the source Azure resource ID.");
  if (input.source.location !== "azure" && input.source.resourceId) throw new Error("Only Azure sources may have an ARM resource ID.");
  return input;
}

export function releaseArtifact(version: string, checksumFile: string): RunnerArtifact {
  if (!/^2\.4\.\d+(?:-[a-z0-9.]+)?$/.test(version)) throw new Error("The runner requires the matching 2.4.x release.");
  const name = `agefreighter_v${version}_linux_amd64.tar.gz`;
  const matches = checksumFile.split(/\r?\n/).flatMap(line => {
    const match = /^([a-f0-9]{64})\s+\*?(?:\.\/)?([^\s]+)$/.exec(line);
    return match?.[2] === name ? [match[1]!] : [];
  });
  if (matches.length !== 1) throw new Error("The matching Linux release checksum is missing or ambiguous. Runner deployment is blocked.");
  return { version, url: `https://github.com/rioriost/agefreighter/releases/download/v${version}/${name}`, sha256: matches[0]! };
}

export function runnerNames(id: string, input: RunnerInput) {
  if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error("Invalid workflow ID.");
  const prefix = `af-${id.replaceAll("-", "").slice(0, 20)}`;
  const base = `/subscriptions/${input.subscriptionId}/resourceGroups/${input.resourceGroup}`;
  return { prefix, deploymentId: `${base}/providers/Microsoft.Resources/deployments/${prefix}`, vmId: `${base}/providers/Microsoft.Compute/virtualMachines/${prefix}` };
}

export function bootstrapScript(artifact: RunnerArtifact): string {
  const validated = releaseArtifact(artifact.version, `${artifact.sha256}  agefreighter_v${artifact.version}_linux_amd64.tar.gz`);
  if (validated.url !== artifact.url) throw new Error("The runner artifact must be the official matching release.");
  return `#!/bin/bash
set -euo pipefail
umask 077
install -d -m 0700 /var/lib/agefreighter /var/lib/agefreighter/evidence
work=$(mktemp -d /var/lib/agefreighter/install.XXXXXX)
curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 --max-time 300 '${artifact.url}' -o "$work/archive.tar.gz"
printf '%s  %s\\n' '${artifact.sha256}' "$work/archive.tar.gz" | sha256sum --check --status
# Extract only the expected executable, not archive-selected paths.
tar -xOzf "$work/archive.tar.gz" agefreighter > "$work/agefreighter"
tar -xOzf "$work/archive.tar.gz" agefreighter-tools > "$work/agefreighter-tools"
install -m 0755 "$work/agefreighter" /usr/local/bin/agefreighter
install -m 0755 "$work/agefreighter-tools" /usr/local/bin/agefreighter-tools
/usr/local/bin/agefreighter version > /var/lib/agefreighter/evidence/version.txt
/usr/local/bin/agefreighter inventory --help > /dev/null
printf '%s\\n' '${artifact.sha256}' > /var/lib/agefreighter/evidence/archive.sha256
touch /var/lib/agefreighter/bootstrap.complete
`;
}

export function runnerTemplate(id: string, input: RunnerInput, artifact: RunnerArtifact, publicKey: string): Record<string, unknown> {
  if (!/^ssh-ed25519 [A-Za-z0-9+/=]+$/.test(publicKey)) throw new Error("Invalid bootstrap public key.");
  const { prefix, vmId } = runnerNames(id, input);
  const base = `/subscriptions/${input.subscriptionId}/resourceGroups/${input.resourceGroup}/providers`;
  const nsgId = `${base}/Microsoft.Network/networkSecurityGroups/${prefix}`;
  const nicId = `${base}/Microsoft.Network/networkInterfaces/${prefix}`;
  const tags = { application: "agefreighter", workflow: id, purpose: "discovery-and-migration" };
  const cloudConfig = "#cloud-config\npackage_update: true\npackages: [curl, ca-certificates]\nwrite_files:\n" +
    "  - path: /var/lib/agefreighter-bootstrap.sh\n    permissions: '0700'\n    encoding: b64\n    content: " +
    Buffer.from(bootstrapScript(artifact)).toString("base64") + "\nruncmd:\n  - [bash, /var/lib/agefreighter-bootstrap.sh]\n";
  return {
    $schema: "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    contentVersion: "2.0.0.0",
    resources: [
      { type: "Microsoft.Network/networkSecurityGroups", apiVersion: "2024-05-01", name: prefix, location: input.region, tags,
        properties: { securityRules: [{ name: "deny-inbound", properties: { priority: 100, direction: "Inbound", access: "Deny", protocol: "*", sourcePortRange: "*", destinationPortRange: "*", sourceAddressPrefix: "*", destinationAddressPrefix: "*" } }] } },
      { type: "Microsoft.Network/networkInterfaces", apiVersion: "2024-05-01", name: prefix, location: input.region, tags, dependsOn: [nsgId],
        properties: { networkSecurityGroup: { id: nsgId }, ipConfigurations: [{ name: "private", properties: { privateIPAllocationMethod: "Dynamic", subnet: { id: input.subnetId } } }] } },
      { type: "Microsoft.Compute/virtualMachines", apiVersion: "2024-07-01", name: prefix, location: input.region, zones: [input.zone], tags, dependsOn: [nicId],
        identity: { type: "SystemAssigned" },
        properties: {
          hardwareProfile: { vmSize: input.size },
          storageProfile: {
            diskControllerType: "SCSI",
            imageReference: { publisher: "Canonical", offer: "0001-com-ubuntu-server-jammy", sku: "22_04-lts-gen2", version: "latest" },
            osDisk: { name: `${prefix}-os`, createOption: "FromImage", diskSizeGB: 64, deleteOption: "Detach", managedDisk: { storageAccountType: "StandardSSD_LRS" } }
          },
          osProfile: { computerName: prefix, adminUsername: "afrunner", customData: Buffer.from(cloudConfig).toString("base64"), linuxConfiguration: { provisionVMAgent: true, disablePasswordAuthentication: true, ssh: { publicKeys: [{ path: "/home/afrunner/.ssh/authorized_keys", keyData: publicKey }] } } },
          networkProfile: { networkInterfaces: [{ id: nicId, properties: { deleteOption: "Detach" } }] }
        }
      }
    ],
    outputs: { vmId: { type: "string", value: vmId } }
  };
}

export function previewHash(template: Record<string, unknown>, input: RunnerInput, hourlyComputeUSD: number): string {
  return createHash("sha256").update(JSON.stringify({ template, input, hourlyComputeUSD })).digest("hex");
}

export function assertFreshPreview(record: RunnerRecord, now = Date.now()): void {
  const names = runnerNames(record.id, parseRunnerInput(record.input));
  if (record.phase !== "previewed" || !Number.isFinite(Date.parse(record.expiresAt)) || now >= Date.parse(record.expiresAt) ||
    record.vmId !== names.vmId || record.deploymentId !== names.deploymentId ||
    !Number.isFinite(record.hourlyComputeUSD) || record.hourlyComputeUSD <= 0 ||
    record.previewHash !== previewHash(record.template, record.input, record.hourlyComputeUSD)) throw new Error("The preview is stale or already submitted. Refresh its status or create a new reviewed preview.");
}

export function validateWhatIf(value: unknown, allowedIds: string[]): void {
  const root = object(value);
  if (root.status !== "Succeeded" || !Array.isArray(root.changes)) throw new Error("Azure what-if did not complete successfully.");
  const expected = new Set(allowedIds.map(id => id.toLowerCase()));
  for (const raw of root.changes) {
    const change = object(raw);
    if (typeof change.resourceId !== "string" || !expected.has(change.resourceId.toLowerCase()) || change.changeType !== "Create") throw new Error("What-if contains an unexpected or existing resource. No deployment is permitted.");
    expected.delete(change.resourceId.toLowerCase());
  }
  if (expected.size) throw new Error("What-if did not prove all expected new resources.");
}

function field(value: Record<string, unknown>, name: string, pattern: RegExp): string {
  const text = value[name];
  if (typeof text !== "string" || text.length > 2048 || /[\u0000-\u001f\u007f]/.test(text) || !pattern.test(text)) throw new Error(`Invalid ${name}.`);
  return text;
}
export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid structured response.");
  return value as Record<string, unknown>;
}
