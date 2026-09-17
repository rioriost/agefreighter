package portable

import (
	"errors"
	"strconv"
	"strings"
)

const GremlinDocumentFormat = "cosmos-gremlin-p1-partition64-v1"

// Derive from the frozen external-ID ordinal, independently of the oracle's
// source-key/first-key arithmetic. Reject malformed IDs rather than hashing them.
func gremlinPartition(label, id string) (string, error) {
	prefix := strings.ToLower(label) + "-"
	if !strings.HasPrefix(id, prefix) || len(id) != len(prefix)+12 {
		return "", errors.New("invalid frozen vertex identity")
	}
	ordinal, err := strconv.ParseInt(id[len(prefix):], 10, 64)
	if err != nil || ordinal < 1 || id[len(prefix):] != leftPadOrdinal(ordinal) {
		return "", errors.New("invalid frozen vertex ordinal")
	}
	return label + "-" + strconv.FormatInt((ordinal-1)%64, 10), nil
}

func leftPadOrdinal(n int64) string {
	s := strconv.FormatInt(n, 10)
	return strings.Repeat("0", 12-len(s)) + s
}

func gremlinDocument(table Table, doc map[string]any) (map[string]any, error) {
	id, ok := doc["id"].(string)
	if !ok || id == "" {
		return nil, errors.New("missing frozen element ID")
	}
	out := map[string]any{"id": id, "label": table.Name}
	if table.Kind == "node" {
		partition, err := gremlinPartition(table.Name, id)
		if err != nil {
			return nil, err
		}
		out["partitionKey"] = partition
	} else if table.Kind == "edge" {
		start, ok1 := doc["start_id"].(string)
		end, ok2 := doc["end_id"].(string)
		if !ok1 || !ok2 {
			return nil, errors.New("missing frozen endpoints")
		}
		partition, err := gremlinPartition(table.StartLabel, start)
		if err != nil {
			return nil, err
		}
		sinkPartition, err := gremlinPartition(table.EndLabel, end)
		if err != nil {
			return nil, err
		}
		out["partitionKey"] = partition
		out["_sinkPartition"] = sinkPartition
		out["_isEdge"] = true
		out["_vertexId"] = start
		out["_vertexLabel"] = table.StartLabel
		out["_sink"] = end
		out["_sinkLabel"] = table.EndLabel
	} else {
		return nil, errors.New("invalid frozen record kind")
	}
	for _, name := range table.Columns {
		if name == "start_id" || name == "end_id" {
			continue
		}
		value, exists := doc[name]
		if !exists {
			return nil, errors.New("missing frozen property")
		}
		if table.Kind == "node" {
			out[name] = []any{map[string]any{"_value": value}}
		} else {
			out[name] = value
		}
	}
	return out, nil
}
