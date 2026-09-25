package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
)

const CatalogMaxTables = 64
const CatalogMaxColumns = 128
const CatalogMaxConstraints = 64
const CatalogMaxBytes = 4 << 20

// Catalog is metadata for an explicit schema scope, never source row counts or
// migration proof. The caller must bind the serialized bytes to its operation.
type Catalog struct {
	SchemaVersion int            `json:"schemaVersion"`
	Command       string         `json:"command"`
	Complete      bool           `json:"complete"`
	Schemas       []string       `json:"schemas"`
	Tables        []CatalogTable `json:"tables"`
}

type CatalogTable struct {
	oid         uint32
	Schema      string              `json:"schema"`
	Name        string              `json:"name"`
	Kind        string              `json:"kind"`
	Readable    bool                `json:"readable"`
	RLS         bool                `json:"rls"`
	Inheritance bool                `json:"inheritance"`
	Columns     []CatalogColumn     `json:"columns"`
	Constraints []CatalogConstraint `json:"constraints"`
}

type CatalogColumn struct {
	Name          string `json:"name"`
	TypeSchema    string `json:"typeSchema"`
	Type          string `json:"type"`
	NotNull       bool   `json:"notNull"`
	Deterministic bool   `json:"deterministic"`
}

type CatalogConstraint struct {
	Name              string   `json:"name"`
	Kind              string   `json:"kind"`
	Columns           []string `json:"columns"`
	ReferencedSchema  string   `json:"referencedSchema"`
	ReferencedTable   string   `json:"referencedTable"`
	ReferencedColumns []string `json:"referencedColumns"`
	Validated         bool     `json:"validated"`
	Enforced          bool     `json:"enforced"`
	Deferrable        bool     `json:"deferrable"`
	Temporal          bool     `json:"temporal"`
	StandardEquality  bool     `json:"standardEquality"`
}

var catalogSchemaName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,62}$`)

func catalogSchemas(schemas []string) ([]string, error) {
	if len(schemas) < 1 || len(schemas) > 16 {
		return nil, errors.New("catalog requires 1-16 explicit schemas")
	}
	result := slices.Clone(schemas)
	slices.Sort(result)
	for i, schema := range result {
		if !catalogSchemaName.MatchString(schema) || schema == "information_schema" || len(schema) >= 3 && schema[:3] == "pg_" || i > 0 && result[i-1] == schema {
			return nil, errors.New("catalog requires unique non-system schema identifiers")
		}
	}
	return result, nil
}

// ReadCatalog reads only pg_catalog using one two-minute, repeatable-read,
// read-only transaction. A restricted search_path and server-side timeouts
// prevent user-schema functions from shadowing catalog operations. The existing
// caller owns DSN/TLS policy; this library never obtains or stores credentials.
func ReadCatalog(ctx context.Context, dsn string, schemas []string) (Catalog, error) {
	scope, err := catalogSchemas(schemas)
	if err != nil {
		return Catalog{}, err
	}
	if ctx == nil || dsn == "" {
		return Catalog{}, errors.New("catalog requires a context and source connection")
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return Catalog{}, safeDatabaseError(ctx, "connect PostgreSQL catalog", err)
	}
	defer func() {
		closeCtx, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		_ = conn.Close(closeCtx)
	}()
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return Catalog{}, safeDatabaseError(ctx, "begin PostgreSQL catalog", err)
	}
	defer func() {
		closeCtx, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		_ = tx.Rollback(closeCtx)
	}()
	if _, err = tx.Exec(ctx, `SET LOCAL search_path = pg_catalog; SET LOCAL statement_timeout = '20s'; SET LOCAL lock_timeout = '2s'; SET LOCAL idle_in_transaction_session_timeout = '30s'`); err != nil {
		return Catalog{}, safeDatabaseError(ctx, "bound PostgreSQL catalog", err)
	}
	return readCatalog(ctx, tx, scope)
}

type catalogQueryer interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

func catalogRows[T any](ctx context.Context, q catalogQueryer, sql string, scan pgx.RowToFunc[T], args ...any) ([]T, error) {
	rows, err := q.Query(ctx, sql, args...)
	if err != nil {
		return nil, safeDatabaseError(ctx, "read PostgreSQL catalog", err)
	}
	result, err := pgx.CollectRows(rows, scan)
	if err != nil {
		return nil, safeDatabaseError(ctx, "decode PostgreSQL catalog", err)
	}
	return result, nil
}

func readCatalog(ctx context.Context, q catalogQueryer, scope []string) (Catalog, error) {
	visible, err := catalogRows(ctx, q, `SELECT nspname::text FROM pg_catalog.pg_namespace WHERE nspname = ANY($1::text[]) AND pg_catalog.has_schema_privilege(oid, 'USAGE') ORDER BY nspname LIMIT 17`, pgx.RowTo[string], scope)
	if err != nil {
		return Catalog{}, err
	}
	slices.Sort(visible)
	if !slices.Equal(visible, scope) {
		return Catalog{}, errors.New("catalog schema is missing or not accessible")
	}
	tables, err := catalogRows(ctx, q, catalogTablesSQL, func(row pgx.CollectableRow) (CatalogTable, error) {
		t := CatalogTable{Columns: []CatalogColumn{}, Constraints: []CatalogConstraint{}}
		err := row.Scan(&t.oid, &t.Schema, &t.Name, &t.Kind, &t.Readable, &t.RLS, &t.Inheritance)
		return t, err
	}, scope, CatalogMaxTables+1)
	if err != nil {
		return Catalog{}, err
	}
	if len(tables) > CatalogMaxTables {
		return Catalog{}, errors.New("catalog exceeds 64 tables; narrow the schema scope")
	}
	for i := range tables {
		t := &tables[i]
		t.Columns, err = catalogRows(ctx, q, catalogColumnsSQL, func(row pgx.CollectableRow) (CatalogColumn, error) {
			var c CatalogColumn
			err := row.Scan(&c.Name, &c.TypeSchema, &c.Type, &c.NotNull, &c.Deterministic)
			return c, err
		}, t.oid, CatalogMaxColumns+1)
		if err != nil {
			return Catalog{}, err
		}
		if len(t.Columns) > CatalogMaxColumns {
			return Catalog{}, errors.New("catalog table exceeds 128 columns")
		}
		t.Constraints, err = catalogRows(ctx, q, catalogConstraintsSQL, func(row pgx.CollectableRow) (CatalogConstraint, error) {
			var c CatalogConstraint
			err := row.Scan(&c.Name, &c.Kind, &c.Columns, &c.ReferencedSchema, &c.ReferencedTable, &c.ReferencedColumns, &c.Validated, &c.Enforced, &c.Deferrable, &c.Temporal, &c.StandardEquality)
			return c, err
		}, t.oid, CatalogMaxConstraints+1)
		if err != nil {
			return Catalog{}, err
		}
		if len(t.Constraints) > CatalogMaxConstraints {
			return Catalog{}, errors.New("catalog table exceeds 64 key constraints")
		}
		if t.Columns == nil {
			t.Columns = []CatalogColumn{}
		}
		if t.Constraints == nil {
			t.Constraints = []CatalogConstraint{}
		}
	}
	if tables == nil {
		tables = []CatalogTable{}
	}
	doc := Catalog{SchemaVersion: 1, Command: "postgres-catalog", Complete: true, Schemas: scope, Tables: tables}
	data, err := json.Marshal(doc)
	if err != nil || len(data) > CatalogMaxBytes {
		return Catalog{}, errors.New("catalog report exceeds its output bound")
	}
	return doc, nil
}

const catalogTablesSQL = `SELECT c.oid, n.nspname::text, c.relname::text, c.relkind::text,
pg_catalog.has_table_privilege(c.oid, 'SELECT'), c.relrowsecurity,
EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid)
FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname = ANY($1::text[]) AND c.relkind IN ('r','p','v','m','f')
ORDER BY n.nspname, c.relname LIMIT $2`

const catalogColumnsSQL = `SELECT a.attname::text, n.nspname::text, t.typname::text, a.attnotnull, COALESCE(coll.collisdeterministic,true)
FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_type t ON t.oid=a.atttypid
JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
LEFT JOIN pg_catalog.pg_collation coll ON coll.oid=a.attcollation
WHERE a.attrelid=$1 AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum LIMIT $2`

// JSON field access keeps pre-18 servers compatible without treating new
// unenforced or temporal constraints as ordinary validated keys.
const catalogConstraintsSQL = `SELECT c.conname::text, c.contype::text,
ARRAY(SELECT a.attname::text FROM unnest(c.conkey) WITH ORDINALITY k(num, ord)
JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.num ORDER BY k.ord),
COALESCE(n.nspname::text,''), COALESCE(r.relname::text,''),
ARRAY(SELECT a.attname::text FROM unnest(c.confkey) WITH ORDINALITY k(num, ord)
JOIN pg_catalog.pg_attribute a ON a.attrelid=c.confrelid AND a.attnum=k.num ORDER BY k.ord),
c.convalidated, COALESCE((to_jsonb(c)->>'conenforced')::boolean,true), c.condeferrable,
COALESCE((to_jsonb(c)->>'conperiod')::boolean,false),
COALESCE((SELECT i.indisvalid AND i.indisready
 AND NOT EXISTS (SELECT 1 FROM unnest(i.indclass) k(oid) JOIN pg_catalog.pg_opclass op ON op.oid=k.oid
  WHERE op.opcnamespace <> 'pg_catalog'::regnamespace OR NOT op.opcdefault)
 AND NOT EXISTS (SELECT 1 FROM unnest(i.indcollation) k(oid) JOIN pg_catalog.pg_collation coll ON coll.oid=k.oid WHERE NOT coll.collisdeterministic)
 FROM pg_catalog.pg_index i WHERE i.indexrelid=c.conindid),false)
 AND NOT EXISTS (SELECT 1 FROM unnest(c.conpfeqop) k(oid) JOIN pg_catalog.pg_operator op ON op.oid=k.oid
  WHERE op.oprnamespace <> 'pg_catalog'::regnamespace OR op.oprname <> '=')
FROM pg_catalog.pg_constraint c LEFT JOIN pg_catalog.pg_class r ON r.oid=c.confrelid
LEFT JOIN pg_catalog.pg_namespace n ON n.oid=r.relnamespace
WHERE c.conrelid=$1 AND c.contype IN ('p','f') ORDER BY c.conname LIMIT $2`
