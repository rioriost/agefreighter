package postgres

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

func validUnitOptions() IteratorOptions {
	return IteratorOptions{
		Namespace: "crm",
		DSN:       "postgres://user:password@localhost:5432/source?sslmode=disable",
		Source: config.PostgreSQLSource{
			ReadMode:  config.PostgreSQLReadCursor,
			FetchRows: 10,
			Vertices: []config.VertexQuery{{
				Label: "Person", Query: "SELECT id FROM people ORDER BY id", IDField: "id",
			}},
		},
	}
}

func fingerprintForOptions(t *testing.T, options IteratorOptions) ([]compiledMapping, string) {
	t.Helper()
	maxProperties := options.MaxProperties
	if maxProperties == 0 {
		maxProperties = 1024
	}
	mappings, err := buildMappings(t.Context(), options.Namespace, options.Source, maxProperties)
	if err != nil {
		t.Fatalf("build mappings: %v", err)
	}
	identity, err := sourceIdentity(options.DSN)
	if err != nil {
		t.Fatalf("source identity: %v", err)
	}
	fingerprint, err := bindFingerprint(
		identity, options.Namespace, options.Source.ReadMode, options.Source.FetchRows, mappings,
	)
	if err != nil {
		t.Fatalf("bind fingerprint: %v", err)
	}
	return mappings, fingerprint
}

func resumeForOptions(
	t *testing.T,
	options IteratorOptions,
	edit func(*resumeState),
) string {
	t.Helper()
	mappings, fingerprint := fingerprintForOptions(t, options)
	state := resumeState{
		fingerprint: fingerprint, mappingIndex: 0, mappingKind: mappings[0].kind,
	}
	if edit != nil {
		edit(&state)
	}
	token, err := formatResumeToken(state)
	if err != nil {
		t.Fatalf("format token: %v", err)
	}
	return token
}

func TestNewIteratorValidation(t *testing.T) {
	valid := validUnitOptions()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewIterator(nil, valid); err == nil {
		t.Fatal("NewIterator(nil) succeeded")
	}
	if _, err := NewIterator(cancelled, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("NewIterator(cancelled) error = %v", err)
	}

	tests := []struct {
		name string
		edit func(*IteratorOptions)
		want string
	}{
		{"mode", func(o *IteratorOptions) { o.Source.ReadMode = "offset" }, "read mode"},
		{"fetch low", func(o *IteratorOptions) { o.Source.FetchRows = 0 }, "fetch rows"},
		{"fetch high", func(o *IteratorOptions) { o.Source.FetchRows = 100_001 }, "fetch rows"},
		{"dsn", func(o *IteratorOptions) { o.DSN = "" }, "DSN"},
		{"reject negative", func(o *IteratorOptions) { o.RejectLimit = -1 }, "reject limit"},
		{"handler", func(o *IteratorOptions) { o.RejectLimit = 1 }, "malformed handler"},
		{"record bytes", func(o *IteratorOptions) { o.MaxRecordBytes = -1 }, "record bytes"},
		{"properties", func(o *IteratorOptions) { o.MaxProperties = -1 }, "properties"},
		{"readers low", func(o *IteratorOptions) { o.MaxReaders = -1 }, "readers"},
		{"readers high", func(o *IteratorOptions) { o.MaxReaders = 257 }, "readers"},
		{"mapping", func(o *IteratorOptions) { o.Namespace = "" }, "namespace"},
		{"dsn parse", func(o *IteratorOptions) { o.DSN = "postgres://%" }, "parse PostgreSQL"},
		{"token parse", func(o *IteratorOptions) { o.AfterToken = "bad" }, "resume token"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := valid
			options.Source.Vertices = append([]config.VertexQuery(nil), valid.Source.Vertices...)
			test.edit(&options)
			_, err := NewIterator(t.Context(), options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewIterator() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestNewIteratorResumeValidation(t *testing.T) {
	base := validUnitOptions()
	withHandler := func(_ context.Context, _ MalformedRecord) error { return nil }
	tests := []struct {
		name  string
		setup func(*IteratorOptions)
		state func(*resumeState)
		want  string
	}{
		{
			"fingerprint", nil,
			func(state *resumeState) { state.fingerprint = strings.Repeat("00", 32) },
			"fingerprint changed",
		},
		{
			"mapping index", nil,
			func(state *resumeState) { state.mappingIndex = 1 },
			"mapping index",
		},
		{
			"mapping kind", nil,
			func(state *resumeState) { state.mappingKind = edgeMapping },
			"mapping kind",
		},
		{
			"rejected", func(options *IteratorOptions) {
				options.RejectLimit = 1
				options.OnMalformed = withHandler
			},
			func(state *resumeState) { state.rejected = 2 },
			"reject limit",
		},
		{
			"non keyset key", nil,
			func(state *resumeState) {
				key, _ := parseNumberKey("1")
				state.key = &key
			},
			"non-keyset",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := base
			options.Source.Vertices = append([]config.VertexQuery(nil), base.Source.Vertices...)
			if test.setup != nil {
				test.setup(&options)
			}
			options.AfterToken = resumeForOptions(t, options, test.state)
			_, err := NewIterator(t.Context(), options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewIterator() error = %v, want %q", err, test.want)
			}
		})
	}

	keyset := base
	keyset.Source.ReadMode = config.PostgreSQLReadKeyset
	keyset.Source.Vertices = []config.VertexQuery{{
		Label: "Person",
		Query: "SELECT id FROM people WHERE ($1::bigint IS NULL OR id > $1) " +
			"ORDER BY id LIMIT $2",
		IDField: "id", KeyField: "id",
	}}
	for _, test := range []struct {
		name string
		edit func(*resumeState)
	}{
		{"consumed without key", func(state *resumeState) { state.consumed = 1 }},
		{"key without consumed", func(state *resumeState) {
			key, _ := parseNumberKey("1")
			state.key = &key
		}},
	} {
		t.Run("keyset "+test.name, func(t *testing.T) {
			options := keyset
			options.AfterToken = resumeForOptions(t, options, test.edit)
			_, err := NewIterator(t.Context(), options)
			if err == nil || !strings.Contains(err.Error(), "key is inconsistent") {
				t.Fatalf("NewIterator() error = %v", err)
			}
		})
	}
}

func encodedResumePayload(t *testing.T, payload resumeTokenPayload) string {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return resumeTokenPrefix + base64.RawURLEncoding.EncodeToString(raw)
}

func TestResumeTokenValidationBranches(t *testing.T) {
	valid := resumeTokenPayload{
		Version: resumeTokenVersion, Fingerprint: strings.Repeat("ab", 32),
		MappingKind: "vertex",
	}
	tests := []struct {
		name  string
		token func() string
		want  string
	}{
		{"too large", func() string { return strings.Repeat("x", maxResumeTokenBytes+1) }, "too large"},
		{"prefix", func() string { return "other:v1:x" }, "unrecognized"},
		{"base64", func() string { return resumeTokenPrefix + "!" }, "base64url"},
		{"payload", func() string {
			return resumeTokenPrefix + base64.RawURLEncoding.EncodeToString([]byte("{"))
		}, "payload"},
		{"trailing", func() string {
			return resumeTokenPrefix + base64.RawURLEncoding.EncodeToString([]byte(
				`{"v":1,"fp":"`+strings.Repeat("ab", 32)+`","mi":0,"mk":"vertex","n":0,"r":0} {}`,
			))
		}, "trailing"},
		{"version", func() string {
			p := valid
			p.Version = 2
			return encodedResumePayload(t, p)
		}, "version 2"},
		{"fingerprint nonhex", func() string {
			p := valid
			p.Fingerprint = strings.Repeat("z", 64)
			return encodedResumePayload(t, p)
		}, "fingerprint"},
		{"fingerprint length", func() string {
			p := valid
			p.Fingerprint = "ab"
			return encodedResumePayload(t, p)
		}, "fingerprint"},
		{"mapping index", func() string {
			p := valid
			p.MappingIndex = -1
			return encodedResumePayload(t, p)
		}, "mapping index"},
		{"mapping kind", func() string {
			p := valid
			p.MappingKind = "document"
			return encodedResumePayload(t, p)
		}, "mapping kind"},
		{"consumed", func() string {
			p := valid
			p.Consumed = -1
			return encodedResumePayload(t, p)
		}, "consumed"},
		{"rejected", func() string {
			p := valid
			p.Rejected = -1
			return encodedResumePayload(t, p)
		}, "rejected"},
		{"key size", func() string {
			p := valid
			p.Key = &resumeKey{Type: keyNumber, Value: strings.Repeat("1", maxResumeKeyBytes+1)}
			return encodedResumePayload(t, p)
		}, "key is too large"},
		{"key type", func() string {
			p := valid
			p.Key = &resumeKey{Type: "unsupported", Value: "1"}
			return encodedResumePayload(t, p)
		}, "key type"},
		{"key value", func() string {
			p := valid
			p.Key = &resumeKey{Type: keyNumber, Value: "1.5"}
			return encodedResumePayload(t, p)
		}, "signed 64-bit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := parseResumeToken(test.token())
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("parseResumeToken() error = %v, want %q", err, test.want)
			}
		})
	}

	longKey := &keyValue{kind: keyNumber, text: strings.Repeat("1", maxResumeKeyBytes+1)}
	if _, err := formatResumeToken(resumeState{key: longKey}); err == nil {
		t.Fatal("formatResumeToken() accepted oversized key")
	}
	edge := valid
	edge.MappingKind = "edge"
	if state, err := parseResumeToken(encodedResumePayload(t, edge)); err != nil ||
		state.mappingKind != edgeMapping {
		t.Fatalf("edge token = %#v, %v", state, err)
	}
}

func TestMappingValidationBranches(t *testing.T) {
	if got := mappingKind(99).String(); got != "unknown" {
		t.Fatalf("unknown mapping kind = %q", got)
	}
	if _, err := buildMappings(t.Context(), "", config.PostgreSQLSource{}, 1); err == nil {
		t.Fatal("buildMappings() accepted empty namespace")
	}
	if _, err := buildMappings(
		t.Context(), "crm", config.PostgreSQLSource{}, 1,
	); err == nil || !strings.Contains(err.Error(), "no mappings") {
		t.Fatalf("empty mappings error = %v", err)
	}

	base := config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCursor,
		Vertices: []config.VertexQuery{{
			Label: "Person", IDField: "id", Query: "SELECT id FROM people ORDER BY id",
		}},
	}
	tests := []struct {
		name string
		edit func(*config.PostgreSQLSource)
		want string
	}{
		{"vertex label", func(s *config.PostgreSQLSource) { s.Vertices[0].Label = "" }, "label"},
		{"vertex id", func(s *config.PostgreSQLSource) { s.Vertices[0].IDField = "" }, "idField"},
		{"order", func(s *config.PostgreSQLSource) {
			s.Vertices[0].Query = "SELECT id FROM people"
		}, "ORDER BY"},
		{"select", func(s *config.PostgreSQLSource) {
			s.Vertices[0].Query = "DELETE FROM people ORDER BY id"
		}, "SELECT or WITH"},
		{"property count", func(s *config.PostgreSQLSource) {
			s.Vertices[0].Properties = map[string]string{"a": "a", "b": "b"}
		}, "maximum"},
		{"property name", func(s *config.PostgreSQLSource) {
			s.Vertices[0].Properties = map[string]string{"": "a"}
		}, "name"},
		{"property field", func(s *config.PostgreSQLSource) {
			s.Vertices[0].Properties = map[string]string{"a": ""}
		}, "source field"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := base
			source.Vertices = append([]config.VertexQuery(nil), base.Vertices...)
			test.edit(&source)
			_, err := buildMappings(t.Context(), "crm", source, 1)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("buildMappings() error = %v, want %q", err, test.want)
			}
		})
	}

	edgeBase := config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCursor,
		Edges: []config.EdgeQuery{{
			Label: "KNOWS", Query: "SELECT s, e FROM edges ORDER BY s",
			Start: config.EndpointMapping{Label: "Person", Field: "s"},
			End:   config.EndpointMapping{Label: "Person", Field: "e"},
		}},
	}
	edgeTests := []struct {
		name string
		edit func(*config.EdgeQuery)
		want string
	}{
		{"label", func(e *config.EdgeQuery) { e.Label = "" }, "label"},
		{"start label", func(e *config.EdgeQuery) { e.Start.Label = "" }, "start label"},
		{"start field", func(e *config.EdgeQuery) { e.Start.Field = "" }, "start field"},
		{"end label", func(e *config.EdgeQuery) { e.End.Label = "" }, "end label"},
		{"end field", func(e *config.EdgeQuery) { e.End.Field = "" }, "end field"},
	}
	for _, test := range edgeTests {
		t.Run("edge "+test.name, func(t *testing.T) {
			source := edgeBase
			source.Edges = append([]config.EdgeQuery(nil), edgeBase.Edges...)
			test.edit(&source.Edges[0])
			_, err := buildMappings(t.Context(), "crm", source, 1)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("buildMappings() error = %v, want %q", err, test.want)
			}
		})
	}
	if err := validateEndpoint(
		config.EndpointMapping{Label: "Person", Field: "id"}, "", "endpoint",
	); err == nil || !strings.Contains(err.Error(), "namespace") {
		t.Fatalf("validateEndpoint() error = %v", err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := buildMappings(cancelled, "crm", edgeBase, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled edge build error = %v", err)
	}
}

func TestValueErrorBranches(t *testing.T) {
	for _, test := range []struct {
		name string
		raw  []byte
		want string
	}{
		{"decode", []byte("{"), "decode"},
		{"null", []byte("null"), "object"},
		{"trailing", []byte(`{} {}`), "trailing"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := decodeObject(test.raw)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("decodeObject() error = %v", err)
			}
		})
	}
	invalidUTF8 := string([]byte{0xff})
	values := []struct {
		name  string
		value any
		depth int
		want  string
	}{
		{"depth", nil, model.MaxPropertyDepth + 1, "nesting"},
		{"string utf8", invalidUTF8, 0, "UTF-8"},
		{"list child", []any{make(chan int)}, 0, "unsupported"},
		{"object name", map[string]any{invalidUTF8: nil}, 0, "name"},
		{"object child", map[string]any{"bad": make(chan int)}, 0, "unsupported"},
		{"unsupported", make(chan int), 0, "unsupported"},
	}
	for _, test := range values {
		t.Run(test.name, func(t *testing.T) {
			_, err := convertValue(test.value, test.depth)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("convertValue() error = %v", err)
			}
		})
	}

	document := map[string]any{
		"empty": "", "badUTF8": invalidUTF8, "decimal": json.Number("1.5"),
		"infinite": json.Number("1e999"), "invalid": json.Number("x"),
		"null": nil, "bool": true, "huge": json.Number("9223372036854775808"),
	}
	for _, test := range []struct {
		field string
		want  string
	}{
		{"missing", "missing"}, {"empty", "empty"}, {"badUTF8", "UTF-8"},
		{"infinite", "finite"}, {"invalid", "integer"}, {"null", "null"},
		{"bool", "string or number"},
	} {
		t.Run("identity "+test.field, func(t *testing.T) {
			_, err := resolveExternalID(document, test.field, "id")
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("resolveExternalID() error = %v, want %q", err, test.want)
			}
		})
	}
	if id, err := resolveExternalID(document, "decimal", "id"); err != nil || id != "1.5" {
		t.Fatalf("decimal identity = %q, %v", id, err)
	}
	if id, err := resolveExternalID(document, "huge", "id"); err != nil ||
		id != "9223372036854775808" {
		t.Fatalf("huge identity = %q, %v", id, err)
	}
}

func TestKeyScalarErrorBranches(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{"row", `{`, "decode"},
		{"missing", `{}`, "missing"},
		{"bad key", `{"id":`, "decode"},
		{"null", `{"id":null}`, "null"},
		{"string", `{"id":"1"}`, "signed 64-bit"},
		{"bool", `{"id":true}`, "signed 64-bit"},
		{"array", `{"id":[1]}`, "signed 64-bit"},
		{"decimal", `{"id":1.5}`, "signed 64-bit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := extractKey([]byte(test.raw), "id")
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("extractKey() error = %v, want %q", err, test.want)
			}
		})
	}
	key, err := extractKey([]byte(`{"id":-2}`), "id")
	if err != nil || key.native != int64(-2) {
		t.Fatalf("extractKey() = %#v, %v", key, err)
	}
	if _, err := parseNumberKey(strings.Repeat("1", maxResumeKeyBytes+1)); err == nil {
		t.Fatal("parseNumberKey() accepted oversized key")
	}
	if _, err := compareKeys(
		keyValue{kind: "bad", native: int64(1)},
		keyValue{kind: "bad", native: int64(2)},
	); err == nil {
		t.Fatal("compareKeys() accepted invalid matching kinds")
	}
	for _, test := range []struct {
		left, right int64
		want        int
	}{
		{1, 2, -1}, {2, 1, 1}, {2, 2, 0},
	} {
		left := keyValue{kind: keyNumber, native: test.left}
		right := keyValue{kind: keyNumber, native: test.right}
		if got, err := compareKeys(left, right); err != nil || got != test.want {
			t.Fatalf("compareKeys(%d,%d) = %d, %v", test.left, test.right, got, err)
		}
	}
}

func TestFingerprintAndTelemetry(t *testing.T) {
	first, err := sourceIdentity(
		"postgres://user:one@host1:5432/database?sslmode=disable",
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := sourceIdentity(
		"postgres://user:two@host1:5432/database?sslmode=disable",
	)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || strings.Contains(first, "one") || strings.Contains(first, "two") {
		t.Fatalf("source identities unexpectedly differ or expose password: %q / %q", first, second)
	}
	withSearchPath, err := sourceIdentity(
		"******host1:5432/database?sslmode=disable&search_path=archive",
	)
	if err != nil {
		t.Fatal(err)
	}
	if first == withSearchPath {
		t.Fatal("source identity did not bind startup session parameters")
	}
	if _, err := sourceIdentity("postgres://%"); err == nil {
		t.Fatal("sourceIdentity() accepted malformed DSN")
	}
	mappings, _ := buildMappings(t.Context(), "crm", validUnitOptions().Source, 10)
	fingerprint, err := bindFingerprint(
		first, "crm", config.PostgreSQLReadCursor, 10, mappings,
	)
	if err != nil || len(fingerprint) != 64 {
		t.Fatalf("bindFingerprint() = %q, %v", fingerprint, err)
	}
	changed, _ := bindFingerprint(
		first, "other", config.PostgreSQLReadCursor, 10, mappings,
	)
	if fingerprint == changed {
		t.Fatal("fingerprint did not bind namespace")
	}

	var telemetry telemetryState
	telemetry.page()
	telemetry.page()
	if err := telemetry.input(7, 11); err != nil {
		t.Fatal(err)
	}
	telemetry.failure()
	got := telemetry.snapshot()
	if got.Connector != "postgresql" || got.Pages != 2 ||
		got.RawInputBytes != 7 || got.DecodedInputBytes != 11 ||
		got.FailedRequestAttempts != 1 {
		t.Fatalf("telemetry = %#v", got)
	}
	iterator := &Iterator{}
	iterator.telemetry.page()
	iterator.telemetry.page()
	if err := iterator.telemetry.input(7, 11); err != nil {
		t.Fatal(err)
	}
	iterator.telemetry.failure()
	if iterator.Telemetry() != got {
		t.Fatalf("Iterator.Telemetry() = %#v", iterator.Telemetry())
	}
}

func TestSizeAndPositionHelpers(t *testing.T) {
	if got := saturatingAdd(math.MaxInt64-1, 10); got != math.MaxInt64 {
		t.Fatalf("saturatingAdd overflow = %d", got)
	}
	if got := saturatingAdd(2, -1); got != 1 {
		t.Fatalf("saturatingAdd negative = %d", got)
	}
	value := model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"s": {Kind: model.ValueString, String: "text"},
		"l": {Kind: model.ValueList, List: []model.Value{{Kind: model.ValueInteger}}},
	}}
	if got := estimateValueSize(value); got <= firstMapBucket {
		t.Fatalf("object estimate = %d", got)
	}
	if got := estimateValueSize(model.Value{Kind: model.ValueObject}); got != propertyBase+mapBase {
		t.Fatalf("empty object estimate = %d", got)
	}
	if got := estimateValueSize(model.Value{Kind: model.ValueBoolean}); got != propertyBase {
		t.Fatalf("scalar estimate = %d", got)
	}

	position := model.SourcePosition{Line: 7}
	vertex := model.VertexRecord(model.Vertex{})
	setPosition(&vertex, position)
	if vertex.Vertex.Position.Line != 7 {
		t.Fatalf("vertex position = %#v", vertex.Vertex.Position)
	}
	edge := model.EdgeRecord(model.Edge{})
	setPosition(&edge, position)
	if edge.Edge.Position.Line != 7 {
		t.Fatalf("edge position = %#v", edge.Edge.Position)
	}
	invalid := model.Record{}
	setPosition(&invalid, position)
}

type readerStep struct {
	row sourceRow
	err error
}

type scriptedRecordReader struct {
	steps    []readerStep
	index    int
	closeErr error
	closed   int
}

func (reader *scriptedRecordReader) Next(context.Context) (sourceRow, error) {
	if reader.index >= len(reader.steps) {
		return sourceRow{}, io.EOF
	}
	step := reader.steps[reader.index]
	reader.index++
	return step.row, step.err
}

func (reader *scriptedRecordReader) Close() error {
	reader.closed++
	return reader.closeErr
}

func inertCoordinator(closeErr error) *SnapshotCoordinator {
	coordinator := &SnapshotCoordinator{err: closeErr}
	coordinator.once.Do(func() {})
	return coordinator
}

func TestIteratorStateAndMalformedBranches(t *testing.T) {
	if _, err := (&Iterator{closed: true}).Next(t.Context()); err == nil {
		t.Fatal("closed iterator Next() succeeded")
	}
	if _, err := (&Iterator{exhausted: true}).Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("exhausted iterator error = %v", err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (&Iterator{}).Next(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Next() error = %v", err)
	}

	mapping := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm", idField: "id",
	}
	iterator := &Iterator{
		options:  IteratorOptions{MaxRecordBytes: 1024},
		mappings: []compiledMapping{mapping}, fingerprint: strings.Repeat("ab", 32),
		current: &scriptedRecordReader{steps: []readerStep{{
			row: sourceRow{raw: []byte(`{"id":"p1"}`)},
		}}},
	}
	item, err := iterator.Next(t.Context())
	if err != nil || item.Record.Vertex.ExternalID != "p1" || item.Record.Vertex.Position.Line != 1 {
		t.Fatalf("Next() = %#v, %v", item, err)
	}
	if rejected, position := iterator.RejectionCheckpoint(); rejected != 0 || position.Line != 1 {
		t.Fatalf("checkpoint = %d, %#v", rejected, position)
	}

	sourceErr := errors.New("source failed")
	iterator = &Iterator{
		mappings: []compiledMapping{mapping},
		current:  &scriptedRecordReader{steps: []readerStep{{err: sourceErr}}},
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, sourceErr) {
		t.Fatalf("source Next() error = %v", err)
	}

	closeErr := errors.New("reader close")
	current := &scriptedRecordReader{
		steps: []readerStep{{err: io.EOF}}, closeErr: closeErr,
	}
	iterator = &Iterator{
		mappings: []compiledMapping{mapping}, current: current,
		coordinator: inertCoordinator(nil),
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("EOF Next() error = %v", err)
	}
	if !errors.Is(iterator.closeErr, closeErr) || current.closed != 1 {
		t.Fatalf("close state = %v, calls %d", iterator.closeErr, current.closed)
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("second EOF Next() error = %v", err)
	}

	recordErr := errors.New("bad record")
	iterator = &Iterator{}
	if err := iterator.handleMalformed(t.Context(), mapping, recordErr); !errors.Is(err, recordErr) {
		t.Fatalf("fail policy error = %v", err)
	}
	iterator = &Iterator{
		options: IteratorOptions{
			RejectLimit: 0,
			OnMalformed: func(context.Context, MalformedRecord) error { return nil },
		},
	}
	if err := iterator.handleMalformed(t.Context(), mapping, recordErr); err == nil ||
		!strings.Contains(err.Error(), "reject limit") {
		t.Fatalf("limit error = %v", err)
	}
	var handled MalformedRecord
	iterator = &Iterator{
		options: IteratorOptions{
			RejectLimit: 2,
			OnMalformed: func(_ context.Context, record MalformedRecord) error {
				handled = record
				return nil
			},
		},
		fingerprint: strings.Repeat("ab", 32), consumed: 4,
	}
	if err := iterator.handleMalformed(t.Context(), mapping, recordErr); err != nil {
		t.Fatalf("handleMalformed() error = %v", err)
	}
	if iterator.rejected != 1 || handled.Position.Line != 4 ||
		!errors.Is(handled.Err, recordErr) {
		t.Fatalf("handled = %#v, rejected %d", handled, iterator.rejected)
	}
	handlerErr := errors.New("quarantine failed")
	iterator.options.OnMalformed = func(context.Context, MalformedRecord) error {
		return handlerErr
	}
	if err := iterator.handleMalformed(t.Context(), mapping, recordErr); !errors.Is(err, handlerErr) {
		t.Fatalf("handler error = %v", err)
	}
	longKey := &keyValue{kind: keyNumber, text: strings.Repeat("1", maxResumeKeyBytes+1)}
	iterator = &Iterator{
		options: IteratorOptions{
			RejectLimit: 1,
			OnMalformed: func(context.Context, MalformedRecord) error { return nil },
		},
		lastKey: longKey,
	}
	if err := iterator.handleMalformed(t.Context(), mapping, recordErr); err == nil ||
		!strings.Contains(err.Error(), "key is too large") {
		t.Fatalf("position error = %v", err)
	}
}

func TestIteratorDecodeEdgeAndErrors(t *testing.T) {
	base := &Iterator{options: IteratorOptions{MaxRecordBytes: 1024}}
	vertex := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm", idField: "id",
		properties: []compiledProperty{{name: "name", field: "name"}},
	}
	if _, _, err := base.decodeRecord(t.Context(), vertex, bytes.Repeat([]byte("x"), 1025)); err == nil {
		t.Fatal("decodeRecord() accepted oversized row")
	}
	if _, _, err := base.decodeRecord(t.Context(), vertex, []byte("{")); err == nil {
		t.Fatal("decodeRecord() accepted invalid JSON")
	}
	if _, _, err := base.decodeRecord(t.Context(), vertex, []byte(`{"id":"p1"}`)); err == nil {
		t.Fatal("decodeRecord() accepted missing property")
	}
	if _, _, err := base.decodeRecord(
		t.Context(), vertex, []byte(`{"id":"p1","name":9223372036854775808}`),
	); err == nil {
		t.Fatal("decodeRecord() accepted invalid property")
	}
	if _, _, err := base.decodeRecord(
		t.Context(), vertex, []byte(`{"name":"Ada"}`),
	); err == nil {
		t.Fatal("decodeRecord() accepted missing vertex identity")
	}

	edge := compiledMapping{
		kind: edgeMapping, label: "KNOWS", namespace: "crm",
		externalIDField: "edge_id",
		start:           config.EndpointMapping{Label: "Person", Field: "source"},
		end: config.EndpointMapping{
			Label: "Person", Namespace: "other", Field: "target",
		},
		properties: []compiledProperty{{name: "weight", field: "weight"}},
	}
	raw := []byte(`{"edge_id":9,"source":"p1","target":"p2","weight":2}`)
	record, size, err := base.decodeRecord(t.Context(), edge, raw)
	if err != nil {
		t.Fatalf("decode edge: %v", err)
	}
	if record.Edge.ExternalID != "9" ||
		record.Edge.Start.Namespace != "crm" ||
		record.Edge.End.Namespace != "other" ||
		size <= edgeBaseSize {
		t.Fatalf("edge = %#v, size %d", record.Edge, size)
	}
	for _, test := range []struct {
		name string
		raw  string
		want string
	}{
		{"external", `{"source":"p1","target":"p2","weight":2}`, "externalIdField"},
		{"start", `{"edge_id":9,"target":"p2","weight":2}`, "start"},
		{"end", `{"edge_id":9,"source":"p1","weight":2}`, "end"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := base.decodeRecord(t.Context(), edge, []byte(test.raw))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("decodeRecord() error = %v", err)
			}
		})
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := base.buildProperties(
		cancelled, map[string]any{"name": "Ada"}, vertex.properties,
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("buildProperties(cancelled) error = %v", err)
	}
	preencode := &Iterator{options: IteratorOptions{PreencodeProperties: true}}
	if _, _, _, err := preencode.buildProperties(
		t.Context(),
		map[string]any{"value": "x"},
		[]compiledProperty{{name: string([]byte{0xff}), field: "value"}},
	); err == nil || !strings.Contains(err.Error(), "encode PostgreSQL") {
		t.Fatalf("preencode invalid name error = %v", err)
	}
}

func TestIteratorCloseBranches(t *testing.T) {
	readerErr := errors.New("reader")
	coordinatorErr := errors.New("coordinator")
	current := &scriptedRecordReader{closeErr: readerErr}
	iterator := &Iterator{
		current: current, coordinator: inertCoordinator(coordinatorErr),
	}
	err := iterator.Close()
	if !errors.Is(err, readerErr) || !errors.Is(err, coordinatorErr) {
		t.Fatalf("Close() error = %v", err)
	}
	if second := iterator.Close(); !errors.Is(second, readerErr) ||
		current.closed != 1 {
		t.Fatalf("second Close() = %v, close calls %d", second, current.closed)
	}
	if err := (&Iterator{}).closeCurrent(); err != nil {
		t.Fatalf("closeCurrent(nil) = %v", err)
	}
}

type fakeRows struct {
	values  []string
	index   int
	scanErr error
	err     error
	closed  bool
}

func (rows *fakeRows) Close()                                       { rows.closed = true }
func (rows *fakeRows) Err() error                                   { return rows.err }
func (rows *fakeRows) CommandTag() pgconn.CommandTag                { return pgconn.CommandTag{} }
func (rows *fakeRows) FieldDescriptions() []pgconn.FieldDescription { return nil }
func (rows *fakeRows) Next() bool {
	if rows.index >= len(rows.values) {
		rows.closed = true
		return false
	}
	rows.index++
	return true
}
func (rows *fakeRows) Scan(dest ...any) error {
	if rows.scanErr != nil {
		return rows.scanErr
	}
	*dest[0].(*string) = rows.values[rows.index-1]
	return nil
}
func (rows *fakeRows) Values() ([]any, error) { return nil, nil }
func (rows *fakeRows) RawValues() [][]byte    { return nil }
func (rows *fakeRows) Conn() *pgx.Conn        { return nil }

func closedSnapshotReader(err error) *SnapshotReader {
	reader := &SnapshotReader{err: err}
	reader.once.Do(func() {})
	return reader
}

func TestCursorReaderPureBranches(t *testing.T) {
	if _, err := (&cursorReader{done: true}).Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("done cursor error = %v", err)
	}
	rows := &fakeRows{values: []string{`{"id":1}`}}
	reader := &cursorReader{rows: rows, fetchRows: 2, telemetry: &telemetryState{}}
	row, err := reader.Next(t.Context())
	if err != nil || string(row.raw) != `{"id":1}` || reader.pageRows != 1 {
		t.Fatalf("cursor row = %q, %v", row.raw, err)
	}
	if _, err := reader.Next(t.Context()); !errors.Is(err, io.EOF) || !reader.done {
		t.Fatalf("cursor final error = %v", err)
	}

	scanErr := errors.New("scan")
	reader = &cursorReader{
		rows:      &fakeRows{values: []string{"x"}, scanErr: scanErr},
		telemetry: &telemetryState{},
	}
	if _, err := reader.Next(t.Context()); err == nil ||
		reader.telemetry.snapshot().FailedRequestAttempts != 1 {
		t.Fatalf("cursor scan error = %v, telemetry %#v", err, reader.telemetry.snapshot())
	}
	reader = &cursorReader{
		rows: &fakeRows{err: errors.New("rows")}, telemetry: &telemetryState{},
	}
	if _, err := reader.Next(t.Context()); err == nil {
		t.Fatal("cursor rows error was ignored")
	}
	reader = &cursorReader{
		rows: &fakeRows{}, fetchRows: 0, telemetry: &telemetryState{},
	}
	if _, err := reader.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("zero-row cursor error = %v", err)
	}

	closeErr := errors.New("close")
	closeRows := &fakeRows{}
	reader = &cursorReader{rows: closeRows, snapshot: closedSnapshotReader(closeErr)}
	if err := reader.Close(); !errors.Is(err, closeErr) || !closeRows.closed {
		t.Fatalf("cursor Close() = %v, rows closed %v", err, closeRows.closed)
	}
	if err := reader.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("second cursor Close() = %v", err)
	}
}

func TestKeysetReaderPureBranches(t *testing.T) {
	if _, err := (&keysetReader{done: true}).Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("done keyset error = %v", err)
	}
	mapping := compiledMapping{keyField: "id"}
	rows := &fakeRows{values: []string{`{"id":2}`}}
	reader := newKeysetReader(
		nil, mapping, 2, nil, &telemetryState{},
	)
	reader.rows = rows
	row, err := reader.Next(t.Context())
	if err != nil || row.key == nil || row.key.native != int64(2) {
		t.Fatalf("keyset row = %#v, %v", row, err)
	}
	if _, err := reader.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("keyset final error = %v", err)
	}

	for _, test := range []struct {
		name     string
		value    string
		previous *keyValue
		want     string
	}{
		{"key", `{}`, nil, "missing keyField"},
		{
			"type", `{"id":2}`,
			&keyValue{kind: "unsupported", native: int64(1)}, "type changed",
		},
		{"equal", `{"id":2}`, &keyValue{kind: keyNumber, native: int64(2)}, "strictly increasing"},
		{"decreasing", `{"id":1}`, &keyValue{kind: keyNumber, native: int64(2)}, "strictly increasing"},
	} {
		t.Run(test.name, func(t *testing.T) {
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{
					Rows: 10, DecodedInputBytes: 1 << 20,
				},
			)
			state := &telemetryState{profileBudget: budget}
			reader := &keysetReader{
				mapping: mapping, fetchRows: 2,
				rows:    &fakeRows{values: []string{test.value}},
				lastKey: test.previous, telemetry: state,
			}
			_, err := reader.Next(t.Context())
			if err == nil || !strings.Contains(err.Error(), test.want) ||
				state.snapshot().FailedRequestAttempts != 1 {
				t.Fatalf("keyset error = %v, telemetry %#v", err, state.snapshot())
			}
			usage, _ := budget.Snapshot()
			if usage.Rows != 1 || usage.DecodedInputBytes != int64(len(test.value)) {
				t.Fatalf("keyset invalid-row usage = %#v", usage)
			}
		})
	}
	reader = &keysetReader{
		rows: &fakeRows{err: errors.New("rows")}, telemetry: &telemetryState{},
	}
	if _, err := reader.Next(t.Context()); err == nil {
		t.Fatal("keyset rows error was ignored")
	}
	reader = &keysetReader{
		rows: &fakeRows{}, fetchRows: 0, telemetry: &telemetryState{},
	}
	if _, err := reader.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("zero-row keyset error = %v", err)
	}

	closeErr := errors.New("close")
	closeRows := &fakeRows{}
	reader = &keysetReader{rows: closeRows, snapshot: closedSnapshotReader(closeErr)}
	if err := reader.Close(); !errors.Is(err, closeErr) || !closeRows.closed {
		t.Fatalf("keyset Close() = %v, rows closed %v", err, closeRows.closed)
	}
	if err := reader.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("second keyset Close() = %v", err)
	}
}

func TestPagedReadersCheckBudgetBeforeQueryAndChargeEmptyPage(t *testing.T) {
	tests := []struct {
		name string
		next func(*telemetryState, func(context.Context, string, ...any) (pgx.Rows, error)) error
	}{
		{
			name: "cursor",
			next: func(telemetry *telemetryState, query func(context.Context, string, ...any) (pgx.Rows, error)) error {
				_, err := (&cursorReader{
					fetchRows: 2, telemetry: telemetry, query: query,
				}).Next(t.Context())
				return err
			},
		},
		{
			name: "keyset",
			next: func(telemetry *telemetryState, query func(context.Context, string, ...any) (pgx.Rows, error)) error {
				_, err := (&keysetReader{
					mapping:   compiledMapping{query: "SELECT 1", keyField: "id"},
					fetchRows: 2, telemetry: telemetry, query: query,
				}).Next(t.Context())
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name+" exhausted", func(t *testing.T) {
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{Pages: 1},
			)
			if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1}); err != nil {
				t.Fatal(err)
			}
			queries := 0
			err := test.next(
				&telemetryState{profileBudget: budget},
				func(context.Context, string, ...any) (pgx.Rows, error) {
					queries++
					return &fakeRows{}, nil
				},
			)
			if !errors.Is(err, sourcecontract.ErrProfileBudget) || queries != 0 {
				t.Fatalf("Next() error = %v, queries = %d", err, queries)
			}
		})
		t.Run(test.name+" empty page", func(t *testing.T) {
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{Pages: 2},
			)
			telemetry := &telemetryState{profileBudget: budget}
			queries := 0
			err := test.next(
				telemetry,
				func(context.Context, string, ...any) (pgx.Rows, error) {
					queries++
					return &fakeRows{}, nil
				},
			)
			usage, _ := budget.Snapshot()
			if !errors.Is(err, io.EOF) || queries != 1 ||
				usage.Pages != 1 || telemetry.snapshot().Pages != 1 {
				t.Fatalf(
					"Next() error = %v, queries = %d, usage = %#v, telemetry = %#v",
					err, queries, usage, telemetry.snapshot(),
				)
			}
		})
	}
}

type failingReader struct {
	err error
}

func (reader failingReader) Read([]byte) (int, error) { return 0, reader.err }

func TestCopyReaderPureBranches(t *testing.T) {
	success := &copyReader{
		scanner: bufio.NewScanner(strings.NewReader(`"{""id"":1}"`)),
		cancel:  func() {},
		pipe:    mustPipeReader(),
		done:    make(chan error, 1),
	}
	row, err := success.Next(t.Context())
	if err != nil || string(row.raw) != `{"id":1}` {
		t.Fatalf("copy row = %q, %v", row.raw, err)
	}

	eof := &copyReader{
		scanner: bufio.NewScanner(strings.NewReader("")),
		cancel:  func() {}, pipe: mustPipeReader(), done: make(chan error, 1),
	}
	eof.done <- nil
	if _, err := eof.Next(t.Context()); !errors.Is(err, io.EOF) || !eof.completed {
		t.Fatalf("copy EOF error = %v", err)
	}
	streamErr := errors.New("stream")
	failed := &copyReader{
		scanner: bufio.NewScanner(failingReader{err: streamErr}),
		cancel:  func() {}, pipe: mustPipeReader(), done: make(chan error, 1),
	}
	if _, err := failed.Next(t.Context()); !errors.Is(err, streamErr) {
		t.Fatalf("copy scanner error = %v", err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	cancelReader := &copyReader{
		scanner: bufio.NewScanner(strings.NewReader("")),
		cancel:  func() {}, pipe: mustPipeReader(), done: make(chan error, 1),
	}
	if _, err := cancelReader.Next(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("copy cancellation error = %v", err)
	}

	if _, err := decodeCopyCSV([]byte("\"unterminated")); err == nil {
		t.Fatal("decodeCopyCSV() accepted invalid CSV")
	}
	if _, err := decodeCopyCSV([]byte("one\nsecond")); err == nil ||
		!strings.Contains(err.Error(), "trailing") {
		t.Fatalf("decodeCopyCSV trailing error = %v", err)
	}
	if got := boundedScannerSize(math.MaxInt64); got != int(^uint(0)>>1) {
		t.Fatalf("boundedScannerSize(max) = %d", got)
	}
	if got := boundedScannerSize(-1000); got != 1 {
		t.Fatalf("boundedScannerSize(-1000) = %d", got)
	}

	closeErr := errors.New("close")
	closeReader := &copyReader{
		cancel: func() {}, pipe: mustPipeReader(), done: make(chan error, 1),
		snapshot: closedSnapshotReader(closeErr),
	}
	closeReader.done <- nil
	if err := closeReader.Close(); !errors.Is(err, closeErr) || !closeReader.completed {
		t.Fatalf("copy Close() = %v", err)
	}
	if err := closeReader.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("second copy Close() = %v", err)
	}
}

func mustPipeReader() *io.PipeReader {
	reader, writer := io.Pipe()
	_ = writer.Close()
	return reader
}

func TestSnapshotValidationAndSafeErrors(t *testing.T) {
	if _, err := NewSnapshotCoordinator(nil, "dsn", 1); err == nil {
		t.Fatal("NewSnapshotCoordinator(nil) succeeded")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewSnapshotCoordinator(cancelled, "dsn", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled coordinator error = %v", err)
	}
	if _, err := NewSnapshotCoordinator(t.Context(), "", 1); err == nil {
		t.Fatal("empty coordinator DSN succeeded")
	}
	for _, readers := range []int{0, 257} {
		if _, err := NewSnapshotCoordinator(t.Context(), "dsn", readers); err == nil {
			t.Fatalf("reader bound %d succeeded", readers)
		}
	}
	if _, err := NewSnapshotCoordinator(t.Context(), "postgres://%", 1); err == nil {
		t.Fatal("malformed coordinator DSN succeeded")
	}

	active, activeCancel := context.WithCancel(context.Background())
	defer activeCancel()
	coordinator := &SnapshotCoordinator{
		ctx: active, slots: make(chan struct{}, 1), readers: map[*SnapshotReader]struct{}{},
	}
	if _, err := coordinator.OpenReader(nil); err == nil {
		t.Fatal("OpenReader(nil) succeeded")
	}
	blocked := &SnapshotCoordinator{
		ctx: active, slots: make(chan struct{}), readers: map[*SnapshotReader]struct{}{},
	}
	if _, err := blocked.OpenReader(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("blocked OpenReader() error = %v", err)
	}
	coordinator.closed = true
	if _, err := coordinator.OpenReader(t.Context()); err == nil ||
		!strings.Contains(err.Error(), "closed") {
		t.Fatalf("closed OpenReader() error = %v", err)
	}
	closedCtx, closeContext := context.WithCancel(context.Background())
	closeContext()
	closedCoordinator := &SnapshotCoordinator{
		ctx: closedCtx, slots: make(chan struct{}), readers: map[*SnapshotReader]struct{}{},
	}
	if _, err := closedCoordinator.OpenReader(t.Context()); err == nil {
		t.Fatal("OpenReader() succeeded after coordinator context closed")
	}

	if err := safeDatabaseError(cancelled, "operation", errors.New("x")); !errors.Is(err, context.Canceled) {
		t.Fatalf("safeDatabaseError(ctx) = %v", err)
	}
	if err := safeDatabaseError(nil, "operation", context.DeadlineExceeded); !errors.Is(
		err, context.DeadlineExceeded,
	) {
		t.Fatalf("safeDatabaseError(deadline) = %v", err)
	}
	pgErr := &pgconn.PgError{Code: "42601", Message: "secret query"}
	if err := safeDatabaseError(nil, "operation", pgErr); !strings.Contains(err.Error(), "42601") ||
		strings.Contains(err.Error(), "secret query") {
		t.Fatalf("safeDatabaseError(pg) = %v", err)
	}
	if err := safeDatabaseError(nil, "operation", errors.New("network detail")); err.Error() !=
		"operation failed" {
		t.Fatalf("safeDatabaseError(generic) = %v", err)
	}
}
