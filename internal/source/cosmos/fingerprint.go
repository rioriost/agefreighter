package cosmos

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// fingerprintParameter is the canonical, order-preserving JSON shape used
// when binding a mapping's query parameters into the source fingerprint.
type fingerprintParameter struct {
	Name  string `json:"name"`
	Value any    `json:"value"`
}

// fingerprintMapping is the canonical JSON shape of one compiled mapping
// used when binding the overall source fingerprint. Every field that can
// change the meaning of the data returned by a mapping is included.
type fingerprintMapping struct {
	Index                int                    `json:"index"`
	Kind                 string                 `json:"kind"`
	Container            string                 `json:"container"`
	Label                string                 `json:"label"`
	Namespace            string                 `json:"namespace"`
	Query                string                 `json:"query"`
	Parameters           []fingerprintParameter `json:"parameters,omitempty"`
	IDField              string                 `json:"idField,omitempty"`
	ExternalIDField      string                 `json:"externalIdField,omitempty"`
	StartLabel           string                 `json:"startLabel,omitempty"`
	StartNamespace       string                 `json:"startNamespace,omitempty"`
	StartField           string                 `json:"startField,omitempty"`
	EndLabel             string                 `json:"endLabel,omitempty"`
	EndNamespace         string                 `json:"endNamespace,omitempty"`
	EndField             string                 `json:"endField,omitempty"`
	Properties           map[string]string      `json:"properties,omitempty"`
	DocumentFormat       string                 `json:"documentFormat,omitempty"`
	PartitionKeyProperty string                 `json:"partitionKeyProperty,omitempty"`
	MaxProperties        int                    `json:"maxProperties,omitempty"`
}

// fingerprintManifest is the canonical JSON shape hashed to produce a
// source's fingerprint. It binds the endpoint, database, namespace, page
// size, and every ordered mapping (including parameters), so that any
// change which could alter what a resume token replays invalidates it.
type fingerprintManifest struct {
	Version   int                  `json:"v"`
	Endpoint  string               `json:"endpoint"`
	Database  string               `json:"database"`
	Namespace string               `json:"namespace"`
	PageSize  int32                `json:"pageSize"`
	Mappings  []fingerprintMapping `json:"mappings"`
}

const fingerprintVersion = 1

// bindFingerprint computes a stable SHA-256 fingerprint (hex-encoded) of the
// source's identity and every ordered, compiled mapping. encoding/json
// marshals map keys in sorted order, which keeps the digest deterministic
// across process restarts for the same configuration.
func bindFingerprint(
	endpoint, database, namespace string,
	pageSize int32,
	mappings []compiledMapping,
) (string, error) {
	manifest := fingerprintManifest{
		Version:   fingerprintVersion,
		Endpoint:  endpoint,
		Database:  database,
		Namespace: namespace,
		PageSize:  pageSize,
		Mappings:  make([]fingerprintMapping, len(mappings)),
	}
	for index, mapping := range mappings {
		entry := fingerprintMapping{
			Index:                index,
			Kind:                 mapping.kind.String(),
			Container:            mapping.container,
			Label:                string(mapping.label),
			Namespace:            string(mapping.namespace),
			Query:                mapping.query,
			DocumentFormat:       string(mapping.documentFormat),
			PartitionKeyProperty: mapping.partitionKeyProperty,
			MaxProperties:        mapping.maxProperties,
		}
		if len(mapping.parameters) > 0 {
			entry.Parameters = make([]fingerprintParameter, len(mapping.parameters))
			for parameterIndex, parameter := range mapping.parameters {
				entry.Parameters[parameterIndex] = fingerprintParameter{
					Name: parameter.Name, Value: parameter.Value,
				}
			}
		}
		switch mapping.kind {
		case vertexMapping:
			entry.IDField = mapping.idField.raw
		case edgeMapping:
			if mapping.hasExternalID {
				entry.ExternalIDField = mapping.externalIDField.raw
			}
			entry.StartLabel = mapping.start.Label
			entry.StartNamespace = mapping.start.Namespace
			entry.StartField = mapping.startField.raw
			entry.EndLabel = mapping.end.Label
			entry.EndNamespace = mapping.end.Namespace
			entry.EndField = mapping.endField.raw
		}
		if len(mapping.properties) > 0 {
			entry.Properties = make(map[string]string, len(mapping.properties))
			for _, property := range mapping.properties {
				entry.Properties[property.name] = property.pointer.raw
			}
		}
		manifest.Mappings[index] = entry
	}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("encode Cosmos source fingerprint: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
