package neo4j

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"
	"unicode/utf8"

	neotypes "github.com/neo4j/neo4j-go-driver/v6/neo4j/dbtype"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func extractKey(record Record, field string) (int64, error) {
	raw, ok := record.Get(field)
	if !ok {
		return 0, errors.New("Neo4j record is missing keyField")
	}
	key, ok := signedInteger(raw)
	if !ok {
		return 0, errors.New("Neo4j keyField must be a signed 64-bit integer")
	}
	return key, nil
}

func signedInteger(raw any) (int64, bool) {
	switch value := raw.(type) {
	case int64:
		return value, true
	case int:
		return int64(value), true
	case int8:
		return int64(value), true
	case int16:
		return int64(value), true
	case int32:
		return int64(value), true
	default:
		return 0, false
	}
}

func resolveExternalID(record Record, field, what string) (model.ExternalID, error) {
	raw, ok := record.Get(field)
	if !ok {
		return "", fmt.Errorf("Neo4j record is missing %s", what)
	}
	switch value := raw.(type) {
	case string:
		if value == "" {
			return "", fmt.Errorf("Neo4j record %s must not be empty", what)
		}
		if !utf8.ValidString(value) {
			return "", fmt.Errorf("Neo4j record %s is not valid UTF-8", what)
		}
		return model.ExternalID(value), nil
	case float64:
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return "", fmt.Errorf("Neo4j record %s must be finite", what)
		}
		return model.ExternalID(strconv.FormatFloat(value, 'g', -1, 64)), nil
	case nil:
		return "", fmt.Errorf("Neo4j record %s must not be null", what)
	default:
		if integer, ok := signedInteger(raw); ok {
			return model.ExternalID(strconv.FormatInt(integer, 10)), nil
		}
		return "", fmt.Errorf("Neo4j record %s must be a string or number", what)
	}
}

func convertValue(
	raw any,
	depth int,
	policy config.Neo4jMultiLabelPolicy,
) (model.Value, error) {
	if depth > model.MaxPropertyDepth {
		return model.Value{}, fmt.Errorf("Neo4j value nesting exceeds %d", model.MaxPropertyDepth)
	}
	switch value := raw.(type) {
	case nil:
		return model.Value{Kind: model.ValueNull}, nil
	case bool:
		return model.Value{Kind: model.ValueBoolean, Boolean: value}, nil
	case int64:
		return model.Value{Kind: model.ValueInteger, Integer: value}, nil
	case int:
		return model.Value{Kind: model.ValueInteger, Integer: int64(value)}, nil
	case int8:
		return model.Value{Kind: model.ValueInteger, Integer: int64(value)}, nil
	case int16:
		return model.Value{Kind: model.ValueInteger, Integer: int64(value)}, nil
	case int32:
		return model.Value{Kind: model.ValueInteger, Integer: int64(value)}, nil
	case float64:
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return model.Value{}, errors.New("Neo4j number must be finite")
		}
		return model.Value{Kind: model.ValueFloat, Float: value}, nil
	case string:
		if !utf8.ValidString(value) {
			return model.Value{}, errors.New("Neo4j string is not valid UTF-8")
		}
		return model.Value{Kind: model.ValueString, String: value}, nil
	case []byte:
		return model.Value{}, errors.New("Neo4j byte arrays are unsupported")
	case []any:
		list := make([]model.Value, len(value))
		for index, item := range value {
			converted, err := convertValue(item, depth+1, policy)
			if err != nil {
				return model.Value{}, err
			}
			list[index] = converted
		}
		return model.Value{Kind: model.ValueList, List: list}, nil
	case map[string]any:
		return convertObject(value, depth, policy)
	case time.Time:
		return stringValue(value.Format(time.RFC3339Nano))
	case neotypes.Date:
		return stringValue(value.String())
	case neotypes.Time:
		return stringValue(value.String())
	case neotypes.LocalDateTime:
		return stringValue(value.String())
	case neotypes.LocalTime:
		return stringValue(value.String())
	case neotypes.Duration:
		return durationValue(value), nil
	case neotypes.Point2D:
		return point2DValue(value)
	case neotypes.Point3D:
		return point3DValue(value)
	case neotypes.Node:
		if policy == config.Neo4jMultiLabelReject && len(value.Labels) > 1 {
			return model.Value{}, errors.New("Neo4j node has multiple labels")
		}
		return convertObject(value.Props, depth, policy)
	case neotypes.Relationship:
		return convertObject(value.Props, depth, policy)
	case neotypes.Path:
		return model.Value{}, errors.New("Neo4j paths are unsupported")
	default:
		return model.Value{}, fmt.Errorf("Neo4j value has unsupported shape %T", raw)
	}
}

func convertObject(
	raw map[string]any,
	depth int,
	policy config.Neo4jMultiLabelPolicy,
) (model.Value, error) {
	object := make(map[string]model.Value, len(raw))
	for name, item := range raw {
		if !utf8.ValidString(name) {
			return model.Value{}, errors.New("Neo4j object name is not valid UTF-8")
		}
		converted, err := convertValue(item, depth+1, policy)
		if err != nil {
			return model.Value{}, err
		}
		object[name] = converted
	}
	return model.Value{Kind: model.ValueObject, Object: object}, nil
}

func stringValue(value string) (model.Value, error) {
	if !utf8.ValidString(value) {
		return model.Value{}, errors.New("Neo4j temporal string is not valid UTF-8")
	}
	return model.Value{Kind: model.ValueString, String: value}, nil
}

func durationValue(value neotypes.Duration) model.Value {
	return model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"months":      {Kind: model.ValueInteger, Integer: value.Months},
		"days":        {Kind: model.ValueInteger, Integer: value.Days},
		"seconds":     {Kind: model.ValueInteger, Integer: value.Seconds},
		"nanoseconds": {Kind: model.ValueInteger, Integer: int64(value.Nanos)},
	}}
}

func point2DValue(value neotypes.Point2D) (model.Value, error) {
	if math.IsNaN(value.X) || math.IsInf(value.X, 0) ||
		math.IsNaN(value.Y) || math.IsInf(value.Y, 0) {
		return model.Value{}, errors.New("Neo4j point coordinates must be finite")
	}
	return model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"srid": {Kind: model.ValueInteger, Integer: int64(value.SpatialRefId)},
		"x":    {Kind: model.ValueFloat, Float: value.X},
		"y":    {Kind: model.ValueFloat, Float: value.Y},
	}}, nil
}

func point3DValue(value neotypes.Point3D) (model.Value, error) {
	if math.IsNaN(value.X) || math.IsInf(value.X, 0) ||
		math.IsNaN(value.Y) || math.IsInf(value.Y, 0) ||
		math.IsNaN(value.Z) || math.IsInf(value.Z, 0) {
		return model.Value{}, errors.New("Neo4j point coordinates must be finite")
	}
	return model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"srid": {Kind: model.ValueInteger, Integer: int64(value.SpatialRefId)},
		"x":    {Kind: model.ValueFloat, Float: value.X},
		"y":    {Kind: model.ValueFloat, Float: value.Y},
		"z":    {Kind: model.ValueFloat, Float: value.Z},
	}}, nil
}

func estimateRecordSize(record Record, limit int64) (int64, error) {
	var size int64
	seen := make(map[string]struct{}, len(record.Keys()))
	for _, key := range record.Keys() {
		if _, duplicate := seen[key]; duplicate {
			return 0, errors.New("Neo4j record has duplicate column names")
		}
		seen[key] = struct{}{}
		size = saturatingAdd(size, int64(len(key)))
		value, ok := record.Get(key)
		if !ok {
			return 0, errors.New("Neo4j record columns are inconsistent")
		}
		size = saturatingAdd(size, estimateRawSize(value, 0, limit-size))
		if size > limit {
			return size, nil
		}
	}
	return size, nil
}

func estimateRawSize(raw any, depth int, remaining int64) int64 {
	if remaining < 0 {
		return 1
	}
	if depth > model.MaxPropertyDepth {
		return remaining + 1
	}
	const scalar = int64(32)
	switch value := raw.(type) {
	case string:
		return saturatingAdd(scalar, int64(len(value)))
	case []byte:
		return saturatingAdd(scalar, int64(len(value)))
	case []any:
		size := scalar
		for _, item := range value {
			size = saturatingAdd(size, estimateRawSize(item, depth+1, remaining-size))
			if size > remaining {
				break
			}
		}
		return size
	case map[string]any:
		return estimateRawMap(value, depth, remaining)
	case neotypes.Node:
		size := saturatingAdd(scalar, int64(len(value.ElementId)))
		for _, label := range value.Labels {
			size = saturatingAdd(size, int64(len(label)))
		}
		return saturatingAdd(
			size, estimateRawMap(value.Props, depth, remaining-size),
		)
	case neotypes.Relationship:
		size := saturatingAdd(scalar, int64(
			len(value.ElementId)+len(value.StartElementId)+
				len(value.EndElementId)+len(value.Type),
		))
		return saturatingAdd(
			size, estimateRawMap(value.Props, depth, remaining-size),
		)
	case neotypes.Path:
		size := scalar
		for _, node := range value.Nodes {
			size = saturatingAdd(size, estimateRawSize(node, depth+1, remaining-size))
			if size > remaining {
				return size
			}
		}
		for _, relationship := range value.Relationships {
			size = saturatingAdd(
				size, estimateRawSize(relationship, depth+1, remaining-size),
			)
			if size > remaining {
				return size
			}
		}
		return size
	default:
		return scalar
	}
}

func estimateRawMap(raw map[string]any, depth int, remaining int64) int64 {
	const base = int64(64)
	size := base
	for name, item := range raw {
		size = saturatingAdd(size, int64(len(name)))
		size = saturatingAdd(size, estimateRawSize(item, depth+1, remaining-size))
		if size > remaining {
			break
		}
	}
	return size
}
