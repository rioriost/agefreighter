package rangedigest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/pkg/model"
)

type rangeBuilder struct {
	rangeRows int64
	leaves    []Leaf
	root      hash.Hash

	kind       string
	name       string
	rangeIndex int
	rows       int64
	startKey   int64
	endKey     int64
	previous   int64
	hasKey     bool
	digest     hash.Hash
	total      int64
}

func newRangeBuilder(rangeRows int64) (*rangeBuilder, error) {
	if rangeRows < 1 || rangeRows > 10_000_000 {
		return nil, errors.New("range rows must be within 1..10000000")
	}
	return &rangeBuilder{rangeRows: rangeRows, root: sha256.New()}, nil
}

func (builder *rangeBuilder) begin(kind, name string) error {
	if builder.digest != nil || kind == "" || name == "" {
		return errors.New("invalid canonical mapping boundary")
	}
	builder.kind = kind
	builder.name = name
	builder.rangeIndex = 0
	builder.rows = 0
	builder.hasKey = false
	builder.digest = sha256.New()
	return nil
}

func (builder *rangeBuilder) add(key int64, canonical []byte) error {
	if builder.digest == nil {
		return errors.New("canonical mapping was not started")
	}
	if builder.hasKey && key <= builder.previous {
		return fmt.Errorf("%s %q source keys are not strictly increasing", builder.kind, builder.name)
	}
	if builder.rows == 0 {
		builder.startKey = key
	}
	builder.previous = key
	builder.endKey = key
	builder.hasKey = true
	builder.rows++
	builder.total++
	_, _ = builder.digest.Write(canonical)
	if builder.rows == builder.rangeRows {
		builder.flush()
	}
	return nil
}

func (builder *rangeBuilder) end() error {
	if builder.digest == nil {
		return errors.New("canonical mapping was not started")
	}
	if builder.rows > 0 {
		builder.flush()
	}
	builder.digest = nil
	builder.kind = ""
	builder.name = ""
	return nil
}

func (builder *rangeBuilder) flush() {
	digest := hex.EncodeToString(builder.digest.Sum(nil))
	leaf := Leaf{
		Kind: builder.kind, Name: builder.name, RangeIndex: builder.rangeIndex,
		StartKey: builder.startKey, EndKey: builder.endKey,
		Rows: builder.rows, SHA256: digest,
	}
	builder.leaves = append(builder.leaves, leaf)
	fmt.Fprintf(
		builder.root,
		"%s\x00%s\x00%d\x00%d\x00%d\x00%d\x00%s\n",
		leaf.Kind,
		leaf.Name,
		leaf.RangeIndex,
		leaf.StartKey,
		leaf.EndKey,
		leaf.Rows,
		leaf.SHA256,
	)
	builder.rangeIndex++
	builder.rows = 0
	builder.hasKey = false
	builder.digest = sha256.New()
}

func (builder *rangeBuilder) result(source, fixtureRoot, graph, jobID string) Manifest {
	return Manifest{
		Version: ManifestVersion, CanonicalVersion: CanonicalVersion,
		Source: source, FixtureRoot: fixtureRoot, Graph: graph, JobID: jobID,
		RangeRows: builder.rangeRows, RecordCount: builder.total,
		Leaves:     builder.leaves,
		RootSHA256: hex.EncodeToString(builder.root.Sum(nil)),
	}
}

func vertexLine(label string, key int64, externalID string, properties []byte) []byte {
	var output bytes.Buffer
	fmt.Fprintf(&output, "v\x00%s\x00%d\x00%s\x00", label, key, externalID)
	output.Write(properties)
	output.WriteByte('\n')
	return output.Bytes()
}

func edgeLine(
	label string,
	key int64,
	externalID string,
	startExternalID string,
	endExternalID string,
	properties []byte,
) []byte {
	var output bytes.Buffer
	fmt.Fprintf(
		&output,
		"e\x00%s\x00%d\x00%s\x00%s\x00%s\x00",
		label,
		key,
		externalID,
		startExternalID,
		endExternalID,
	)
	output.Write(properties)
	output.WriteByte('\n')
	return output.Bytes()
}

func canonicalJSONProperties(raw string) (model.Properties, []byte, error) {
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()
	var decoded map[string]any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, nil, fmt.Errorf("decode AGE properties: %w", err)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, nil, err
	}
	properties := make(model.Properties, len(decoded))
	for name, value := range decoded {
		converted, err := jsonModelValue(value, 0)
		if err != nil {
			return nil, nil, fmt.Errorf("AGE property %q: %w", name, err)
		}
		properties[name] = converted
	}
	encoded, err := model.EncodeProperties(properties)
	if err != nil {
		return nil, nil, fmt.Errorf("canonicalize AGE properties: %w", err)
	}
	return properties, encoded, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("AGE properties contain multiple JSON values")
	}
	return fmt.Errorf("finish AGE properties: %w", err)
}

func jsonModelValue(raw any, depth int) (model.Value, error) {
	if depth > model.MaxPropertyDepth {
		return model.Value{}, errors.New("property depth exceeds canonical limit")
	}
	switch value := raw.(type) {
	case nil:
		return model.Value{Kind: model.ValueNull}, nil
	case bool:
		return model.Value{Kind: model.ValueBoolean, Boolean: value}, nil
	case string:
		return model.Value{Kind: model.ValueString, String: value}, nil
	case json.Number:
		if strings.ContainsAny(string(value), ".eE") {
			parsed, err := strconv.ParseFloat(string(value), 64)
			if err != nil {
				return model.Value{}, fmt.Errorf("parse float: %w", err)
			}
			return model.Value{Kind: model.ValueFloat, Float: parsed}, nil
		}
		parsed, err := strconv.ParseInt(string(value), 10, 64)
		if err != nil {
			return model.Value{}, fmt.Errorf("parse integer: %w", err)
		}
		return model.Value{Kind: model.ValueInteger, Integer: parsed}, nil
	case []any:
		items := make([]model.Value, len(value))
		for index, item := range value {
			converted, err := jsonModelValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			items[index] = converted
		}
		return model.Value{Kind: model.ValueList, List: items}, nil
	case map[string]any:
		items := make(map[string]model.Value, len(value))
		for name, item := range value {
			converted, err := jsonModelValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			items[name] = converted
		}
		return model.Value{Kind: model.ValueObject, Object: items}, nil
	default:
		return model.Value{}, fmt.Errorf("unsupported JSON value %T", raw)
	}
}

func integerProperty(properties model.Properties, name string) (int64, error) {
	value, ok := properties[name]
	if !ok || value.Kind != model.ValueInteger {
		return 0, fmt.Errorf("property %q is not an integer", name)
	}
	return value.Integer, nil
}

func stringProperty(properties model.Properties, name string) (string, error) {
	value, ok := properties[name]
	if !ok || value.Kind != model.ValueString || value.String == "" {
		return "", fmt.Errorf("property %q is not a non-empty string", name)
	}
	return value.String, nil
}
