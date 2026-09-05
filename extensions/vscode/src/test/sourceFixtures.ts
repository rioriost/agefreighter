import { SourceForm } from "../core/runnerSource";
export const workflow = "11111111-1111-4111-8111-111111111111";
export const csvFile = { id: "22222222-2222-4222-8222-222222222222", name: "vertices.csv" };
export const sourceForm: SourceForm = {
  name: "graph-migration", namespace: "migration", host: "source.example.com", port: 7687, database: "source", username: "reader", vertexKey: "source_key", edgeKey: "source_key",
  cosmosFormat: "explicit", container: "graph", partitionKey: "partitionKey", labelField: "label", nullValue: "\\N",
  mappings: [{ kind: "vertex", label: "Person", collection: "people", schema: "public", identity: "id", startLabel: "", startField: "", endLabel: "", endField: "", properties: "name=full_name,age=age" },
    { kind: "edge", label: "KNOWS", collection: "knows", schema: "public", identity: "id", startLabel: "Person", startField: "from_id", endLabel: "Person", endField: "to_id", properties: "" }]
};
