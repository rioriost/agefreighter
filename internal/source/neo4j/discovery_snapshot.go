package neo4j

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
)

const (
	discoverySnapshotVersion  = 1
	maxDiscoverySnapshotBytes = 4 << 20
)

// DiscoverySnapshot is an operator-reviewed copy of the schema facts normally
// obtained by Neo4j discovery. It allows a durable job to resume without
// rescanning a very large, immutable source merely to reconstruct mappings.
// The iterator fingerprint still binds every generated query and rejects a
// snapshot that does not reproduce the original mapping exactly.
type DiscoverySnapshot struct {
	SchemaVersion int                             `json:"schemaVersion"`
	SourceID      string                          `json:"sourceId"`
	Labels        []DiscoverySnapshotLabel        `json:"labels"`
	Relationships []DiscoverySnapshotRelationship `json:"relationships"`
}

type DiscoverySnapshotLabel struct {
	Name       string   `json:"name"`
	Properties []string `json:"properties"`
}

type DiscoverySnapshotRelationship struct {
	Type       string   `json:"type"`
	StartLabel string   `json:"startLabel"`
	EndLabel   string   `json:"endLabel"`
	Properties []string `json:"properties"`
}

func LoadDiscoverySnapshot(path string) (DiscoverySnapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		return DiscoverySnapshot{}, fmt.Errorf("open Neo4j discovery snapshot: %w", err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxDiscoverySnapshotBytes+1))
	if err != nil {
		return DiscoverySnapshot{}, fmt.Errorf("read Neo4j discovery snapshot: %w", err)
	}
	if len(data) > maxDiscoverySnapshotBytes {
		return DiscoverySnapshot{}, fmt.Errorf(
			"Neo4j discovery snapshot exceeds %d bytes",
			maxDiscoverySnapshotBytes,
		)
	}
	return ParseDiscoverySnapshot(data)
}

func ParseDiscoverySnapshot(data []byte) (DiscoverySnapshot, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var snapshot DiscoverySnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return DiscoverySnapshot{}, fmt.Errorf("decode Neo4j discovery snapshot: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return DiscoverySnapshot{}, errors.New(
				"Neo4j discovery snapshot must contain exactly one JSON value",
			)
		}
		return DiscoverySnapshot{}, fmt.Errorf(
			"decode trailing Neo4j discovery snapshot data: %w",
			err,
		)
	}
	return snapshot, nil
}

// ResolveMappingsSnapshot reconstructs the same deterministic mappings as
// live discovery from an independently reviewed schema snapshot.
func ResolveMappingsSnapshot(
	source config.Neo4jSource,
	snapshot DiscoverySnapshot,
) (config.Neo4jSource, error) {
	if source.Discovery == nil || !source.Discovery.Enabled {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery configuration is required for a snapshot",
		)
	}
	if snapshot.SchemaVersion != discoverySnapshotVersion {
		return config.Neo4jSource{}, fmt.Errorf(
			"Neo4j discovery snapshot schema version %d is not supported",
			snapshot.SchemaVersion,
		)
	}
	if snapshot.SourceID == "" || snapshot.SourceID != source.SourceID {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery snapshot sourceId does not match the job",
		)
	}
	options := *source.Discovery
	if len(snapshot.Labels) == 0 {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery snapshot contains no labels",
		)
	}
	if len(snapshot.Labels) > options.MaxLabels {
		return config.Neo4jSource{}, fmt.Errorf(
			"Neo4j discovery snapshot contains %d labels, maximum is %d",
			len(snapshot.Labels), options.MaxLabels,
		)
	}

	labelsByName := make(map[string][]string, len(snapshot.Labels))
	for _, label := range snapshot.Labels {
		if !validDiscoveredIdentifier(label.Name) ||
			!strings.HasPrefix(label.Name, options.LabelPrefix) {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot contains invalid label %q",
				label.Name,
			)
		}
		if _, exists := labelsByName[label.Name]; exists {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot contains duplicate label %q",
				label.Name,
			)
		}
		properties, err := normalizeSnapshotProperties(
			label.Properties,
			options.MaxProperties,
			"label "+label.Name,
		)
		if err != nil {
			return config.Neo4jSource{}, err
		}
		requiredVertexProperties := []string{options.VertexKeyProperty}
		if options.VertexIdentity != config.Neo4jVertexIdentityInternalID {
			requiredVertexProperties = append(
				requiredVertexProperties,
				options.VertexIDProperty,
			)
		}
		if err := requireProperties(properties, requiredVertexProperties...); err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot label %q: %w",
				label.Name,
				err,
			)
		}
		labelsByName[label.Name] = properties
	}

	labelNames := make([]string, 0, len(labelsByName))
	for name := range labelsByName {
		labelNames = append(labelNames, name)
	}
	slices.Sort(labelNames)
	labels := make([]discoveredLabel, len(labelNames))
	vertices := make([]config.VertexQuery, len(labelNames))
	for index, name := range labelNames {
		labels[index] = discoveredLabel{source: name, target: name}
	}
	for index, label := range labels {
		vertices[index] = buildDiscoveredVertex(
			label,
			labels,
			labelsByName[label.source],
			options,
		)
	}

	relationships := slices.Clone(snapshot.Relationships)
	slices.SortFunc(relationships, func(left, right DiscoverySnapshotRelationship) int {
		if value := strings.Compare(left.Type, right.Type); value != 0 {
			return value
		}
		if value := strings.Compare(left.StartLabel, right.StartLabel); value != 0 {
			return value
		}
		return strings.Compare(left.EndLabel, right.EndLabel)
	})
	edges := make([]config.EdgeQuery, 0, len(relationships))
	relationshipTypeCounts := make(map[string]int, len(relationships))
	for _, relationship := range relationships {
		relationshipTypeCounts[relationship.Type]++
	}
	seenRelationships := make(map[string]struct{}, len(relationships))
	for _, relationship := range relationships {
		if !validDiscoveredIdentifier(relationship.Type) ||
			!strings.HasPrefix(relationship.Type, options.RelationshipTypePrefix) {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot contains invalid relationship type %q",
				relationship.Type,
			)
		}
		if _, ok := labelsByName[relationship.StartLabel]; !ok {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot relationship %q has unknown start label %q",
				relationship.Type,
				relationship.StartLabel,
			)
		}
		if _, ok := labelsByName[relationship.EndLabel]; !ok {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot relationship %q has unknown end label %q",
				relationship.Type,
				relationship.EndLabel,
			)
		}
		identity := relationship.Type + "\x00" + relationship.StartLabel +
			"\x00" + relationship.EndLabel
		if _, exists := seenRelationships[identity]; exists {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot contains duplicate relationship %q",
				identity,
			)
		}
		seenRelationships[identity] = struct{}{}
		properties, err := normalizeSnapshotProperties(
			relationship.Properties,
			options.MaxProperties,
			"relationship "+relationship.Type,
		)
		if err != nil {
			return config.Neo4jSource{}, err
		}
		if err := requireProperties(
			properties,
			options.EdgeKeyProperty,
			options.EdgeIDProperty,
		); err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot relationship %q: %w",
				relationship.Type,
				err,
			)
		}
		edges = append(edges, buildDiscoveredEdge(
			relationship.Type,
			endpointPair{start: relationship.StartLabel, end: relationship.EndLabel},
			labels,
			properties,
			options,
			relationshipTypeCounts[relationship.Type] > 1,
		))
		if len(vertices)+len(edges) > maxDiscoveryMappings {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j discovery snapshot exceeds %d generated mappings",
				maxDiscoveryMappings,
			)
		}
	}

	source.Discovery = nil
	source.Vertices = vertices
	source.Edges = edges
	return source, nil
}

func normalizeSnapshotProperties(
	properties []string,
	maximum int,
	resource string,
) ([]string, error) {
	if len(properties) > maximum {
		return nil, fmt.Errorf(
			"Neo4j discovery snapshot %s contains %d properties, maximum is %d",
			resource,
			len(properties),
			maximum,
		)
	}
	result := slices.Clone(properties)
	for _, property := range result {
		if !validDiscoveredIdentifier(property) {
			return nil, fmt.Errorf(
				"Neo4j discovery snapshot %s contains invalid property %q",
				resource,
				property,
			)
		}
	}
	slices.Sort(result)
	if len(slices.Compact(slices.Clone(result))) != len(result) {
		return nil, fmt.Errorf(
			"Neo4j discovery snapshot %s contains duplicate properties",
			resource,
		)
	}
	return result, nil
}
