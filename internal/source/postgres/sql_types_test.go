package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestLegacyPostgreSQLCheckpointRejectedBeforeConnection(t *testing.T) {
	dsn := "postgres://reader@127.0.0.1:1/source?sslmode=disable"
	identity, err := sourceIdentity(dsn)
	if err != nil {
		t.Fatal(err)
	}
	query := "SELECT id, score FROM supplier ORDER BY id"
	legacy := fingerprintManifest{Version: 1, SourceIdentity: identity, Namespace: "test", ReadMode: config.PostgreSQLReadCopy, FetchRows: 1,
		Mappings: []fingerprintMapping{{Index: 0, Kind: "vertex", KindIndex: 0, Label: "Supplier", Namespace: "test", Query: query, IDField: "id", Properties: []fingerprintProperty{{Name: "score", Field: "score"}}}},
	}
	raw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	token, err := formatResumeToken(resumeState{fingerprint: hex.EncodeToString(sum[:]), mappingKind: vertexMapping, consumed: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewIterator(t.Context(), IteratorOptions{Namespace: "test", DSN: dsn, AfterToken: token, Source: config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCopy, FetchRows: 1, Vertices: []config.VertexQuery{{Label: "Supplier", Query: query, IDField: "id", Properties: map[string]string{"score": "score"}}},
	}})
	if err == nil || !strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("legacy resume error = %v", err)
	}
}

func TestConvertSQLFloatValue(t *testing.T) {
	for _, input := range []string{"74", "0", "-0", "-74", "0.25", "1e20", "9223372036854775808"} {
		value, err := convertSQLValue(json.Number(input), sqlFloat, 0)
		if err != nil || value.Kind != model.ValueFloat {
			t.Fatalf("float %s = %+v, %v", input, value, err)
		}
	}
	for _, raw := range []any{"NaN", "Infinity", "-Infinity", true, json.Number("1e9999"), json.Number("NaN"), json.Number("Infinity")} {
		if _, err := convertSQLValue(raw, sqlFloat, 0); err == nil {
			t.Fatalf("accepted non-finite or non-numeric float %v", raw)
		}
	}
	null, err := convertSQLValue(nil, sqlFloat, 0)
	if err != nil || null.Kind != model.ValueNull {
		t.Fatalf("null = %+v, %v", null, err)
	}
	integer, err := convertSQLValue(json.Number("9223372036854775807"), sqlNotFloat, 0)
	if err != nil || integer.Kind != model.ValueInteger || integer.Integer != 9223372036854775807 {
		t.Fatalf("int64 precision lost: %+v, %v", integer, err)
	}
	array, err := convertSQLValue([]any{[]any{json.Number("74"), nil}}, sqlFloatArray, 0)
	if err != nil || array.List[0].List[0].Kind != model.ValueFloat || array.List[0].List[1].Kind != model.ValueNull {
		t.Fatalf("array = %+v, %v", array, err)
	}
	for _, raw := range []any{json.Number("1"), []any{"NaN"}} {
		if _, err := convertSQLValue(raw, sqlFloatArray, 0); err == nil {
			t.Fatalf("accepted invalid float array %v", raw)
		}
	}
	if _, err := convertSQLValue([]any{}, sqlFloatArray, model.MaxPropertyDepth+1); err == nil {
		t.Fatal("accepted excessive array depth")
	}
}

func TestPostgreSQLNativeFloatTypesIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	domain := pgx.Identifier{fmt.Sprintf("af_float_%d", time.Now().UnixNano())}.Sanitize()
	if _, err := conn.Exec(t.Context(), "CREATE DOMAIN "+domain+" AS double precision"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = conn.Exec(context.Background(), "DROP DOMAIN "+domain) })
	for _, mode := range []config.PostgreSQLReadMode{config.PostgreSQLReadCopy, config.PostgreSQLReadCursor, config.PostgreSQLReadKeyset} {
		for _, preencode := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/preencode=%t", mode, preencode), func(t *testing.T) {
				query := `SELECT seq::bigint AS id, 74::double precision AS score,
  38::real AS real_score, 1e20::double precision AS large_float,
  '-0'::double precision AS zero, NULL::double precision AS missing,
  ARRAY[[74,NULL],[0,38]]::double precision[] AS float_array,
  ARRAY[74,0]::real[] AS real_array,
  9223372036854775807::bigint AS big_integer,
  '-9223372036854775808'::bigint AS small_integer,
  '{"number":74,"fraction":0.25,"list":[1,2]}'::jsonb AS payload,
  74::` + domain + ` AS domain_float,
  ARRAY[74,0]::` + domain + `[] AS domain_array
FROM generate_series(1,2) AS seq`
				keyField := ""
				if mode == config.PostgreSQLReadKeyset {
					query += " WHERE ($1::bigint IS NULL OR seq > $1) ORDER BY seq LIMIT $2"
					keyField = "id"
				} else {
					query += " ORDER BY seq"
				}
				properties := map[string]string{}
				for _, name := range []string{"score", "real_score", "large_float", "zero", "missing", "float_array", "real_array", "big_integer", "small_integer", "payload", "domain_float", "domain_array"} {
					properties[name] = name
				}
				source := config.PostgreSQLSource{ReadMode: mode, FetchRows: 1, Vertices: []config.VertexQuery{{
					Label: "Supplier", Query: query, KeyField: keyField, IDField: "id", Properties: properties,
				}}}
				iterator, err := NewIterator(t.Context(), IteratorOptions{Namespace: "test", Source: source, DSN: dsn, PreencodeProperties: preencode})
				if err != nil {
					t.Fatal(err)
				}
				defer iterator.Close()
				for index := 1; index <= 2; index++ {
					item, err := iterator.Next(t.Context())
					if err != nil {
						t.Fatal(err)
					}
					if item.Record.Vertex.ExternalID != model.ExternalID(fmt.Sprint(index)) {
						t.Fatal("identity spelling changed")
					}
					encoded := item.Record.Vertex.EncodedProperties
					if !preencode {
						encoded, err = model.EncodeProperties(item.Record.Vertex.Properties)
						if err != nil {
							t.Fatal(err)
						}
					}
					for _, expected := range []string{`"score":74.0`, `"real_score":38.0`, `"zero":-0.0`, `"missing":null`, `"float_array":[[74.0,null],[0.0,38.0]]`, `"real_array":[74.0,0.0]`, `"domain_float":74.0`, `"domain_array":[74.0,0.0]`, `"big_integer":9223372036854775807`, `"small_integer":-9223372036854775808`, `"payload":{"fraction":0.25,"list":[1,2],"number":74}`} {
						if !strings.Contains(string(encoded), expected) {
							t.Fatalf("missing %s in %s", expected, encoded)
						}
					}
				}
				if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
					t.Fatalf("expected EOF: %v", err)
				}
			})
		}
	}
}

func TestPostgreSQLNonFiniteFloatsIntegration(t *testing.T) {
	_, dsn := integrationConnection(t)
	for _, mode := range []config.PostgreSQLReadMode{config.PostgreSQLReadCopy, config.PostgreSQLReadCursor, config.PostgreSQLReadKeyset} {
		for _, literal := range []string{"NaN", "Infinity", "-Infinity"} {
			query := "SELECT 1::bigint AS id, '" + literal + "'::double precision AS score"
			key := ""
			if mode == config.PostgreSQLReadKeyset {
				query += " WHERE ($1::bigint IS NULL OR 1 > $1) ORDER BY id LIMIT $2"
				key = "id"
			} else {
				query += " ORDER BY id"
			}
			iterator, err := NewIterator(t.Context(), IteratorOptions{Namespace: "test", DSN: dsn, Source: config.PostgreSQLSource{
				ReadMode: mode, FetchRows: 1, Vertices: []config.VertexQuery{{Label: "Supplier", Query: query, IDField: "id", KeyField: key, Properties: map[string]string{"score": "score"}}},
			}})
			if err != nil {
				t.Fatal(err)
			}
			_, err = iterator.Next(t.Context())
			_ = iterator.Close()
			if err == nil || !strings.Contains(err.Error(), "finite number") {
				t.Fatalf("%s/%s: %v", mode, literal, err)
			}
		}
	}
}
