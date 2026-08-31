package neo4j

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/rioriost/agefreighter/internal/config"
)

const fingerprintVersion = 2

type fingerprintProperty struct {
	Name  string `json:"name"`
	Field string `json:"field"`
}

type fingerprintMapping struct {
	Index           int                    `json:"index"`
	Kind            string                 `json:"kind"`
	KindIndex       int                    `json:"kindIndex"`
	Label           string                 `json:"label"`
	Namespace       string                 `json:"namespace"`
	Query           string                 `json:"query"`
	InitialQuery    string                 `json:"initialQuery,omitempty"`
	KeyField        string                 `json:"keyField"`
	IDField         string                 `json:"idField,omitempty"`
	ExternalIDField string                 `json:"externalIdField,omitempty"`
	Start           config.EndpointMapping `json:"start,omitempty"`
	End             config.EndpointMapping `json:"end,omitempty"`
	Properties      []fingerprintProperty  `json:"properties,omitempty"`
}

type fingerprintManifest struct {
	Version          int                          `json:"v"`
	SourceID         string                       `json:"sourceId"`
	URI              string                       `json:"uri"`
	Database         string                       `json:"database"`
	Username         string                       `json:"username"`
	FetchRows        int                          `json:"fetchRows"`
	MultiLabelPolicy config.Neo4jMultiLabelPolicy `json:"multiLabelPolicy"`
	Namespace        string                       `json:"namespace"`
	Mappings         []fingerprintMapping         `json:"mappings"`
}

func bindFingerprint(
	source config.Neo4jSource,
	namespace string,
	mappings []compiledMapping,
) (string, error) {
	manifest := fingerprintManifest{
		Version: fingerprintVersion, SourceID: source.SourceID, URI: source.URI,
		Database: source.Database, Username: source.Username, FetchRows: source.FetchRows,
		MultiLabelPolicy: source.MultiLabelPolicy, Namespace: namespace,
		Mappings: make([]fingerprintMapping, len(mappings)),
	}
	for index, mapping := range mappings {
		entry := fingerprintMapping{
			Index: index, Kind: mapping.kind.String(), KindIndex: mapping.kindIndex,
			Label: string(mapping.label), Namespace: string(mapping.namespace),
			Query: mapping.query, InitialQuery: mapping.initialQuery,
			KeyField: mapping.keyField, IDField: mapping.idField,
			ExternalIDField: mapping.externalIDField, Start: mapping.start, End: mapping.end,
			Properties: make([]fingerprintProperty, len(mapping.properties)),
		}
		for propertyIndex, property := range mapping.properties {
			entry.Properties[propertyIndex] = fingerprintProperty{
				Name: property.name, Field: property.field,
			}
		}
		manifest.Mappings[index] = entry
	}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("encode Neo4j source fingerprint: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
