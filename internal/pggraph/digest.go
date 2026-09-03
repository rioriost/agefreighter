package pggraph

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"slices"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/meta"
)

const DigestRangeCount = 256

var ErrIntegrity = errors.New("PostgreSQL property graph integrity check failed")

type DigestSet struct {
	Root   string
	Rows   int64
	Ranges []meta.PropertyGraphDigestRange
}

type canonicalRecord struct {
	Kind           string          `json:"kind"`
	Label          string          `json:"label"`
	Namespace      string          `json:"namespace"`
	ExternalID     string          `json:"externalId"`
	StartLabel     string          `json:"startLabel,omitempty"`
	StartNamespace string          `json:"startNamespace,omitempty"`
	StartID        string          `json:"startId,omitempty"`
	EndLabel       string          `json:"endLabel,omitempty"`
	EndNamespace   string          `json:"endNamespace,omitempty"`
	EndID          string          `json:"endId,omitempty"`
	Properties     json.RawMessage `json:"properties"`
}

func vertexRecordDigest(
	label string,
	namespace string,
	externalID string,
	properties []byte,
) (uint8, string, error) {
	return logicalRecordDigest(canonicalRecord{
		Kind: "v", Label: label, Namespace: namespace,
		ExternalID: externalID, Properties: properties,
	})
}

func edgeRecordDigest(
	label string,
	namespace string,
	externalID string,
	startLabel string,
	startNamespace string,
	startID string,
	endLabel string,
	endNamespace string,
	endID string,
	properties []byte,
) (uint8, string, error) {
	return logicalRecordDigest(canonicalRecord{
		Kind: "e", Label: label, Namespace: namespace,
		ExternalID: externalID, StartLabel: startLabel,
		StartNamespace: startNamespace, StartID: startID,
		EndLabel: endLabel, EndNamespace: endNamespace, EndID: endID,
		Properties: properties,
	})
}

func logicalRecordDigest(record canonicalRecord) (uint8, string, error) {
	properties, err := canonicalJSON(record.Properties)
	if err != nil {
		return 0, "", err
	}
	record.Properties = properties
	identity, err := json.Marshal(struct {
		Kind       string `json:"kind"`
		Label      string `json:"label"`
		Namespace  string `json:"namespace"`
		ExternalID string `json:"externalId"`
	}{record.Kind, record.Label, record.Namespace, record.ExternalID})
	if err != nil {
		return 0, "", fmt.Errorf("encode logical identity: %w", err)
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		return 0, "", fmt.Errorf("encode logical record: %w", err)
	}
	rangeDigest := sha256.Sum256(identity)
	recordDigest := sha256.Sum256(encoded)
	return rangeDigest[0], hex.EncodeToString(recordDigest[:]), nil
}

func canonicalJSON(value []byte) ([]byte, error) {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode properties: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("properties must contain one JSON value")
	}
	object, ok := decoded.(map[string]any)
	if !ok || object == nil {
		return nil, errors.New("properties must be a JSON object")
	}
	normalized, err := normalizeJSONValue(object)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(normalized)
	if err != nil {
		return nil, fmt.Errorf("encode canonical properties: %w", err)
	}
	return encoded, nil
}

// normalizeJSONValue mirrors PostgreSQL jsonb's numeric rendering so a source
// spelling such as 1e0 hashes identically to the target value rendered as 1.
func normalizeJSONValue(value any) (any, error) {
	switch typed := value.(type) {
	case json.Number:
		normalized, err := normalizePostgreSQLNumeric(typed.String())
		if err != nil {
			return nil, err
		}
		return json.Number(normalized), nil
	case []any:
		for index, item := range typed {
			normalized, err := normalizeJSONValue(item)
			if err != nil {
				return nil, err
			}
			typed[index] = normalized
		}
	case map[string]any:
		for key, item := range typed {
			normalized, err := normalizeJSONValue(item)
			if err != nil {
				return nil, err
			}
			typed[key] = normalized
		}
	}
	return value, nil
}

func normalizePostgreSQLNumeric(value string) (string, error) {
	negative := strings.HasPrefix(value, "-")
	if negative {
		value = strings.TrimPrefix(value, "-")
	}
	mantissa, exponentText, hasExponent := value, "", false
	if index := strings.IndexAny(value, "eE"); index >= 0 {
		mantissa, exponentText, hasExponent = value[:index], value[index+1:], true
	}
	exponent := int64(0)
	if hasExponent {
		parsed, err := strconv.ParseInt(exponentText, 10, 32)
		if err != nil {
			return "", fmt.Errorf("normalize PostgreSQL numeric exponent: %w", err)
		}
		exponent = parsed
	}
	integer, fraction := mantissa, ""
	if index := strings.IndexByte(mantissa, '.'); index >= 0 {
		integer, fraction = mantissa[:index], mantissa[index+1:]
	}
	digits := integer + fraction
	decimalPosition := int64(len(integer)) + exponent
	scale := int64(len(digits)) - decimalPosition
	if decimalPosition > 131_072 || scale > 16_383 {
		return "", errors.New("JSON number exceeds PostgreSQL numeric limits")
	}
	var normalized string
	switch {
	case decimalPosition <= 0:
		normalized = "0." + strings.Repeat("0", int(-decimalPosition)) + digits
	case decimalPosition >= int64(len(digits)):
		normalized = digits + strings.Repeat("0", int(decimalPosition)-len(digits))
	default:
		normalized = digits[:decimalPosition] + "." + digits[decimalPosition:]
	}
	parts := strings.SplitN(normalized, ".", 2)
	parts[0] = strings.TrimLeft(parts[0], "0")
	if parts[0] == "" {
		parts[0] = "0"
	}
	normalized = strings.Join(parts, ".")
	if negative && strings.Trim(strings.ReplaceAll(normalized, ".", ""), "0") != "" {
		normalized = "-" + normalized
	}
	return normalized, nil
}

type digestQuerier interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

func computeDigests(
	ctx context.Context,
	querier digestQuerier,
	jobID string,
	definition Definition,
) (DigestSet, error) {
	definition = definition.normalized()
	result := DigestSet{}
	for _, vertex := range definition.Vertices {
		ranges, rows, err := digestVertexTable(ctx, querier, jobID, definition.Schema, vertex)
		if err != nil {
			return DigestSet{}, err
		}
		result.Ranges = append(result.Ranges, ranges...)
		result.Rows += rows
	}
	for _, edge := range definition.Edges {
		ranges, rows, err := digestEdgeTable(ctx, querier, jobID, definition.Schema, edge, definition)
		if err != nil {
			return DigestSet{}, err
		}
		result.Ranges = append(result.Ranges, ranges...)
		result.Rows += rows
	}
	slices.SortFunc(result.Ranges, compareDigestRanges)
	root := sha256.New()
	for _, value := range result.Ranges {
		encoded, err := json.Marshal(struct {
			Kind    string `json:"kind"`
			Label   string `json:"label"`
			RangeID uint8  `json:"rangeId"`
			Rows    int64  `json:"rows"`
			Digest  string `json:"digest"`
		}{string(value.Kind), value.LabelName, value.RangeID, value.Rows, value.Digest})
		if err != nil {
			return DigestSet{}, fmt.Errorf("encode digest range: %w", err)
		}
		_, _ = root.Write(encoded)
		_, _ = root.Write([]byte{'\n'})
	}
	result.Root = hex.EncodeToString(root.Sum(nil))
	return result, nil
}

func digestVertexTable(
	ctx context.Context,
	querier digestQuerier,
	jobID string,
	schema string,
	vertex VertexDefinition,
) ([]meta.PropertyGraphDigestRange, int64, error) {
	rows, err := querier.Query(ctx, fmt.Sprintf(`
		SELECT digest_range, source_namespace, external_id, properties, source_digest::text
		FROM %s
		ORDER BY digest_range, source_namespace, external_id`,
		qualifiedName(schema, vertex.Table)))
	if err != nil {
		return nil, 0, fmt.Errorf("query vertex digest records for %q: %w", vertex.Label, err)
	}
	defer rows.Close()
	return collectDigestRows(rows, jobID, vertex.Label, meta.VertexLabel,
		func(rows pgx.Rows) (int, string, string, error) {
			var rangeID int
			var namespace, externalID, sourceDigest string
			var properties []byte
			if err := rows.Scan(&rangeID, &namespace, &externalID, &properties, &sourceDigest); err != nil {
				return 0, "", "", err
			}
			computedRange, computed, err := vertexRecordDigest(
				vertex.Label, namespace, externalID, properties)
			if err != nil {
				return 0, "", "", err
			}
			return rangeID, sourceDigest, computed, validateStoredRange(rangeID, computedRange)
		})
}

func digestEdgeTable(
	ctx context.Context,
	querier digestQuerier,
	jobID string,
	schema string,
	edge EdgeDefinition,
	definition Definition,
) ([]meta.PropertyGraphDigestRange, int64, error) {
	labels := make(map[string]string, len(definition.Vertices))
	for _, vertex := range definition.Vertices {
		labels[vertex.Table] = vertex.Label
	}
	rows, err := querier.Query(ctx, fmt.Sprintf(`
		SELECT edge.digest_range, edge.source_namespace, edge.external_id,
		       source.source_namespace, source.external_id,
		       destination.source_namespace, destination.external_id,
		       edge.properties, edge.source_digest::text
		FROM %s edge
		LEFT JOIN %s source ON source.id = edge.start_id
		LEFT JOIN %s destination ON destination.id = edge.end_id
		ORDER BY edge.digest_range, edge.source_namespace, edge.external_id`,
		qualifiedName(schema, edge.Table),
		qualifiedName(schema, edge.SourceTable),
		qualifiedName(schema, edge.DestinationTable)))
	if err != nil {
		return nil, 0, fmt.Errorf("query edge digest records for %q: %w", edge.Label, err)
	}
	defer rows.Close()
	return collectDigestRows(rows, jobID, edge.Label, meta.EdgeLabel,
		func(rows pgx.Rows) (int, string, string, error) {
			var rangeID int
			var namespace, externalID, startNamespace, startID string
			var endNamespace, endID, sourceDigest string
			var properties []byte
			if err := rows.Scan(
				&rangeID, &namespace, &externalID, &startNamespace, &startID,
				&endNamespace, &endID, &properties, &sourceDigest,
			); err != nil {
				return 0, "", "", err
			}
			computedRange, computed, err := edgeRecordDigest(
				edge.Label, namespace, externalID,
				labels[edge.SourceTable], startNamespace, startID,
				labels[edge.DestinationTable], endNamespace, endID, properties,
			)
			if err != nil {
				return 0, "", "", err
			}
			return rangeID, sourceDigest, computed, validateStoredRange(rangeID, computedRange)
		})
}

type scanDigestRow func(pgx.Rows) (rangeID int, sourceDigest, computed string, err error)

func collectDigestRows(
	rows pgx.Rows,
	jobID string,
	label string,
	kind meta.LabelKind,
	scan scanDigestRow,
) ([]meta.PropertyGraphDigestRange, int64, error) {
	var result []meta.PropertyGraphDigestRange
	var total int64
	currentRange := -1
	var currentRows int64
	var current hash.Hash
	flush := func() {
		if current == nil {
			return
		}
		result = append(result, meta.PropertyGraphDigestRange{
			JobID: jobID, LabelName: label, Kind: kind,
			RangeID: uint8(currentRange), Rows: currentRows,
			Digest: hex.EncodeToString(current.Sum(nil)),
		})
	}
	for rows.Next() {
		rangeID, sourceDigest, computed, err := scan(rows)
		if err != nil {
			return nil, 0, fmt.Errorf("scan logical digest record for %q: %w", label, err)
		}
		if sourceDigest != computed {
			return nil, 0, fmt.Errorf(
				"%w: logical record digest mismatch for %q in range %d",
				ErrIntegrity, label, rangeID)
		}
		if rangeID != currentRange {
			flush()
			currentRange, currentRows, current = rangeID, 0, sha256.New()
		}
		_, _ = current.Write([]byte(computed))
		_, _ = current.Write([]byte{'\n'})
		currentRows++
		total++
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("read logical digest records for %q: %w", label, err)
	}
	flush()
	return result, total, nil
}

func validateStoredRange(stored int, computed uint8) error {
	if stored < 0 || stored >= DigestRangeCount || stored != int(computed) {
		return fmt.Errorf("%w: stored digest range %d does not match computed range %d",
			ErrIntegrity, stored, computed)
	}
	return nil
}

func compareDigestRanges(left, right meta.PropertyGraphDigestRange) int {
	if left.Kind != right.Kind {
		return int(right.Kind) - int(left.Kind)
	}
	if compared := strings.Compare(left.LabelName, right.LabelName); compared != 0 {
		return compared
	}
	return int(left.RangeID) - int(right.RangeID)
}

func CompareDigests(
	expectedRoot string,
	expectedRows int64,
	expected []meta.PropertyGraphDigestRange,
	actual DigestSet,
) error {
	if expectedRoot == "" {
		return fmt.Errorf("%w: persisted property graph digest root is unavailable", ErrIntegrity)
	}
	if expectedRoot != actual.Root || expectedRows != actual.Rows {
		return fmt.Errorf(
			"%w: property graph digest root or row count changed: root=%s rows=%d, expected root=%s rows=%d",
			ErrIntegrity,
			actual.Root, actual.Rows, expectedRoot, expectedRows,
		)
	}
	if len(expected) != len(actual.Ranges) {
		return fmt.Errorf("%w: property graph digest range count is %d, expected %d",
			ErrIntegrity,
			len(actual.Ranges), len(expected))
	}
	ordered := slices.Clone(expected)
	slices.SortFunc(ordered, compareDigestRanges)
	for index := range ordered {
		if ordered[index].LabelName != actual.Ranges[index].LabelName ||
			ordered[index].Kind != actual.Ranges[index].Kind ||
			ordered[index].RangeID != actual.Ranges[index].RangeID ||
			ordered[index].Rows != actual.Ranges[index].Rows ||
			ordered[index].Digest != actual.Ranges[index].Digest {
			return fmt.Errorf("%w: property graph digest range %d changed", ErrIntegrity, index)
		}
	}
	return nil
}
