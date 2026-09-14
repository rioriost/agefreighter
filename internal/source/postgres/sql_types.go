package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strconv"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/rioriost/agefreighter/pkg/model"
)

type sqlFloatKind uint8

const (
	sqlNotFloat sqlFloatKind = iota
	sqlFloat
	sqlFloatArray
)

// Describe the original projection, not row_to_json's single text column.
// Parse/Describe does not execute the source query and works with keyset
// parameters. The reader has already imported the shared read-only snapshot.
func describeFloatFields(ctx context.Context, reader *SnapshotReader, mapping compiledMapping) (map[string]sqlFloatKind, error) {
	description, err := reader.tx.Prepare(ctx, "", mapping.query)
	if err != nil {
		return nil, safeDatabaseError(ctx, "describe PostgreSQL source types", err)
	}
	wanted := make(map[string]bool, len(mapping.properties))
	for _, property := range mapping.properties {
		wanted[property.field] = true
	}
	fields := make(map[string]sqlFloatKind)
	seen := make(map[string]bool)
	cache := make(map[uint32]sqlFloatKind)
	for _, field := range description.Fields {
		if !wanted[field.Name] {
			continue
		}
		if seen[field.Name] {
			return nil, errors.New("PostgreSQL mapped property column names must be unique")
		}
		seen[field.Name] = true
		kind, ok := cache[field.DataTypeOID]
		if !ok {
			kind, err = resolveSQLFloatKind(ctx, reader, field.DataTypeOID, 0)
			if err != nil {
				return nil, err
			}
			cache[field.DataTypeOID] = kind
		}
		if kind != sqlNotFloat {
			fields[field.Name] = kind
		}
	}
	return fields, nil
}

func resolveSQLFloatKind(ctx context.Context, reader *SnapshotReader, oid uint32, depth int) (sqlFloatKind, error) {
	if depth > model.MaxPropertyDepth {
		return sqlNotFloat, errors.New("PostgreSQL SQL type nesting exceeds property depth limit")
	}
	switch oid {
	case pgtype.Float4OID, pgtype.Float8OID:
		return sqlFloat, nil
	case pgtype.Float4ArrayOID, pgtype.Float8ArrayOID:
		return sqlFloatArray, nil
	}
	// Built-in non-floating types retain the existing JSON conversion contract.
	if oid < 16384 {
		return sqlNotFloat, nil
	}
	var base, element uint32
	if err := reader.tx.QueryRow(ctx,
		"SELECT typbasetype, typelem FROM pg_catalog.pg_type WHERE oid = $1", oid,
	).Scan(&base, &element); err != nil {
		return sqlNotFloat, safeDatabaseError(ctx, "resolve PostgreSQL source type", err)
	}
	if base != 0 { // Domains, including domains over arrays.
		return resolveSQLFloatKind(ctx, reader, base, depth+1)
	}
	if element != 0 { // Arrays of domains; dimensions are carried by the value.
		kind, err := resolveSQLFloatKind(ctx, reader, element, depth+1)
		if err != nil {
			return sqlNotFloat, err
		}
		if kind != sqlNotFloat {
			return sqlFloatArray, nil
		}
	}
	return sqlNotFloat, nil
}

func convertSQLValue(raw any, kind sqlFloatKind, depth int) (model.Value, error) {
	if kind == sqlNotFloat || raw == nil {
		return convertValue(raw, depth)
	}
	if depth > model.MaxPropertyDepth {
		return model.Value{}, errors.New("PostgreSQL floating-point array exceeds property depth limit")
	}
	if kind == sqlFloatArray {
		items, ok := raw.([]any)
		if !ok {
			return model.Value{}, errors.New("PostgreSQL floating-point array has invalid shape")
		}
		values := make([]model.Value, len(items))
		for index, item := range items {
			elementKind := sqlFloat
			if _, nested := item.([]any); nested {
				elementKind = sqlFloatArray
			}
			value, err := convertSQLValue(item, elementKind, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			values[index] = value
		}
		return model.Value{Kind: model.ValueList, List: values}, nil
	}
	number, ok := raw.(json.Number)
	if !ok {
		// PostgreSQL serializes NaN/Infinity as JSON strings. Reject these
		// rather than silently changing a floating-point property to text.
		return model.Value{}, errors.New("PostgreSQL floating-point property must be a finite number")
	}
	value, err := strconv.ParseFloat(number.String(), 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return model.Value{}, errors.New("PostgreSQL floating-point property must be a finite number")
	}
	return model.Value{Kind: model.ValueFloat, Float: value}, nil
}
