package postgres

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
)

const fingerprintVersion = 1

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
	KeyField        string                 `json:"keyField,omitempty"`
	IDField         string                 `json:"idField,omitempty"`
	ExternalIDField string                 `json:"externalIdField,omitempty"`
	Start           config.EndpointMapping `json:"start,omitempty"`
	End             config.EndpointMapping `json:"end,omitempty"`
	Properties      []fingerprintProperty  `json:"properties,omitempty"`
}

type fingerprintManifest struct {
	Version        int                       `json:"v"`
	SourceIdentity string                    `json:"sourceIdentity"`
	Namespace      string                    `json:"namespace"`
	ReadMode       config.PostgreSQLReadMode `json:"readMode"`
	FetchRows      int                       `json:"fetchRows"`
	Mappings       []fingerprintMapping      `json:"mappings"`
}

type fingerprintEndpoint struct {
	Host string `json:"host"`
	Port uint16 `json:"port"`
}

type fingerprintParameter struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type fingerprintSource struct {
	Database   string                 `json:"database"`
	User       string                 `json:"user"`
	Endpoints  []fingerprintEndpoint  `json:"endpoints"`
	Parameters []fingerprintParameter `json:"parameters,omitempty"`
}

func sourceIdentity(dsn string) (string, error) {
	connection, err := pgx.ParseConfig(dsn)
	if err != nil {
		return "", errors.New("parse PostgreSQL source connection")
	}
	source := fingerprintSource{
		Database: connection.Database,
		User:     connection.User,
		Endpoints: []fingerprintEndpoint{{
			Host: connection.Host,
			Port: connection.Port,
		}},
	}
	for _, fallback := range connection.Fallbacks {
		source.Endpoints = append(source.Endpoints, fingerprintEndpoint{
			Host: fallback.Host,
			Port: fallback.Port,
		})
	}
	names := make([]string, 0, len(connection.RuntimeParams))
	for name := range connection.RuntimeParams {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		source.Parameters = append(source.Parameters, fingerprintParameter{
			Name:  name,
			Value: connection.RuntimeParams[name],
		})
	}
	encoded, err := json.Marshal(source)
	if err != nil {
		return "", errors.New("encode PostgreSQL source identity")
	}
	return string(encoded), nil
}

func bindFingerprint(
	sourceIdentity string,
	namespace string,
	mode config.PostgreSQLReadMode,
	fetchRows int,
	mappings []compiledMapping,
) (string, error) {
	manifest := fingerprintManifest{
		Version: fingerprintVersion, SourceIdentity: sourceIdentity,
		Namespace: namespace,
		ReadMode:  mode, FetchRows: fetchRows,
		Mappings: make([]fingerprintMapping, len(mappings)),
	}
	for index, mapping := range mappings {
		entry := fingerprintMapping{
			Index: index, Kind: mapping.kind.String(), KindIndex: mapping.kindIndex,
			Label: string(mapping.label), Namespace: string(mapping.namespace),
			Query: mapping.query, KeyField: mapping.keyField,
			IDField: mapping.idField, ExternalIDField: mapping.externalIDField,
			Start: mapping.start, End: mapping.end,
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
		return "", fmt.Errorf("encode PostgreSQL source fingerprint: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
