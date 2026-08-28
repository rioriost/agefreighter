package postgres

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

// MalformedRecord describes a source row that could not be mapped. It never
// contains the source row itself.
type MalformedRecord struct {
	Position model.SourcePosition
	Err      error
}

type MalformedHandler func(context.Context, MalformedRecord) error

// IteratorOptions configures a PostgreSQL iterator.
type IteratorOptions struct {
	Namespace           string
	Source              config.PostgreSQLSource
	DSN                 string
	AfterToken          string
	RejectLimit         int
	MaxRecordBytes      int64
	MaxProperties       int
	OnMalformed         MalformedHandler
	PreencodeProperties bool
	MaxReaders          int
	ProfileBudget       *sourcecontract.ProfileBudget
}

// Iterator reads all configured vertex mappings followed by edge mappings
// from one exported, repeatable-read PostgreSQL snapshot.
type Iterator struct {
	mu          sync.Mutex
	options     IteratorOptions
	mappings    []compiledMapping
	fingerprint string
	coordinator *SnapshotCoordinator
	current     recordReader

	mappingIndex int
	consumed     int64
	lastKey      *keyValue
	resume       resumeState
	hasResume    bool
	rejected     int
	lastPosition model.SourcePosition

	telemetry telemetryState
	closed    bool
	exhausted bool
	closeOnce sync.Once
	closeErr  error
}

func NewIterator(ctx context.Context, options IteratorOptions) (*Iterator, error) {
	if ctx == nil {
		return nil, errors.New("PostgreSQL iterator context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch options.Source.ReadMode {
	case config.PostgreSQLReadCopy,
		config.PostgreSQLReadCursor,
		config.PostgreSQLReadKeyset:
	default:
		return nil, errors.New("PostgreSQL read mode must be copy, cursor, or keyset")
	}
	if options.Source.FetchRows < 1 || options.Source.FetchRows > 100_000 {
		return nil, errors.New("PostgreSQL fetch rows must be between 1 and 100000")
	}
	if options.DSN == "" {
		return nil, errors.New("PostgreSQL DSN is required")
	}
	if options.RejectLimit < 0 {
		return nil, errors.New("PostgreSQL reject limit cannot be negative")
	}
	if options.RejectLimit > 0 && options.OnMalformed == nil {
		return nil, errors.New(
			"PostgreSQL malformed handler is required when reject limit is positive",
		)
	}
	if options.MaxRecordBytes == 0 {
		options.MaxRecordBytes = 16 << 20
	}
	if options.MaxRecordBytes < 1 {
		return nil, errors.New("PostgreSQL maximum record bytes must be positive")
	}
	if options.MaxProperties == 0 {
		options.MaxProperties = 1024
	}
	if options.MaxProperties < 1 {
		return nil, errors.New("PostgreSQL maximum properties must be positive")
	}
	if options.MaxReaders == 0 {
		options.MaxReaders = 1
	}
	if options.MaxReaders < 1 || options.MaxReaders > 256 {
		return nil, errors.New("PostgreSQL maximum readers must be between 1 and 256")
	}

	mappings, err := buildMappings(
		ctx, options.Namespace, options.Source, options.MaxProperties,
	)
	if err != nil {
		return nil, err
	}
	identity, err := sourceIdentity(options.DSN)
	if err != nil {
		return nil, err
	}
	fingerprint, err := bindFingerprint(
		identity,
		options.Namespace,
		options.Source.ReadMode,
		options.Source.FetchRows,
		mappings,
	)
	if err != nil {
		return nil, err
	}
	iterator := &Iterator{
		options: options, mappings: mappings, fingerprint: fingerprint,
	}
	iterator.telemetry.profileBudget = options.ProfileBudget
	if options.AfterToken != "" {
		resume, err := parseResumeToken(options.AfterToken)
		if err != nil {
			return nil, err
		}
		if resume.fingerprint != fingerprint {
			return nil, errors.New("PostgreSQL source fingerprint changed since checkpoint")
		}
		if resume.mappingIndex >= len(mappings) {
			return nil, errors.New("PostgreSQL resume token mapping index is out of range")
		}
		if resume.mappingKind != mappings[resume.mappingIndex].kind {
			return nil, errors.New(
				"PostgreSQL resume token mapping kind does not match configuration",
			)
		}
		if resume.rejected > options.RejectLimit {
			return nil, errors.New("PostgreSQL resume token exceeds configured reject limit")
		}
		if options.Source.ReadMode == config.PostgreSQLReadKeyset {
			if (resume.consumed > 0) != (resume.key != nil) {
				return nil, errors.New("PostgreSQL keyset resume token key is inconsistent")
			}
		} else if resume.key != nil {
			return nil, errors.New("PostgreSQL non-keyset resume token contains a key")
		}
		iterator.resume = resume
		iterator.hasResume = true
		iterator.mappingIndex = resume.mappingIndex
		iterator.consumed = resume.consumed
		iterator.lastKey = resume.key
		iterator.rejected = resume.rejected
		iterator.lastPosition.Token = options.AfterToken
	}
	coordinator, err := NewSnapshotCoordinator(ctx, options.DSN, options.MaxReaders)
	if err != nil {
		return nil, err
	}
	iterator.coordinator = coordinator
	return iterator, nil
}

func (iterator *Iterator) Next(ctx context.Context) (sourcecontract.Item, error) {
	iterator.mu.Lock()
	defer iterator.mu.Unlock()
	if iterator.closed {
		return sourcecontract.Item{}, errors.New("PostgreSQL iterator is closed")
	}
	if iterator.exhausted {
		return sourcecontract.Item{}, io.EOF
	}
	for {
		if err := ctx.Err(); err != nil {
			return sourcecontract.Item{}, err
		}
		if err := iterator.options.ProfileBudget.CanProcess(); err != nil {
			return sourcecontract.Item{}, err
		}
		if iterator.current == nil {
			if iterator.mappingIndex >= len(iterator.mappings) {
				iterator.exhausted = true
				_ = iterator.closeResources()
				return sourcecontract.Item{}, io.EOF
			}
			if err := iterator.openCurrent(ctx); err != nil {
				return sourcecontract.Item{}, err
			}
		}

		row, err := iterator.current.Next(ctx)
		if errors.Is(err, io.EOF) {
			iterator.closeErr = errors.Join(iterator.closeErr, iterator.closeCurrent())
			iterator.mappingIndex++
			iterator.consumed = 0
			iterator.lastKey = nil
			continue
		}
		if err != nil {
			return sourcecontract.Item{}, err
		}
		if !row.accounted {
			if err := iterator.telemetry.input(0, int64(len(row.raw))); err != nil {
				return sourcecontract.Item{}, err
			}
		}
		iterator.consumed++
		if row.key != nil {
			iterator.lastKey = row.key
		}
		mapping := iterator.mappings[iterator.mappingIndex]
		record, size, err := iterator.decodeRecord(ctx, mapping, row.raw)
		if err != nil {
			if handledErr := iterator.handleMalformed(ctx, mapping, err); handledErr != nil {
				return sourcecontract.Item{}, handledErr
			}
			continue
		}
		position, err := iterator.buildPosition(mapping, iterator.rejected)
		if err != nil {
			return sourcecontract.Item{}, err
		}
		setPosition(&record, position)
		iterator.lastPosition = position
		return sourcecontract.Item{Record: record, SizeBytes: size}, nil
	}
}

func (iterator *Iterator) openCurrent(ctx context.Context) error {
	if err := iterator.options.ProfileBudget.Full(); err != nil {
		return err
	}
	mapping := iterator.mappings[iterator.mappingIndex]
	var afterKey *keyValue
	if iterator.hasResume &&
		iterator.options.Source.ReadMode == config.PostgreSQLReadKeyset {
		afterKey = iterator.resume.key
	}
	current, err := openRecordReader(
		ctx,
		iterator.coordinator,
		mapping,
		string(iterator.options.Source.ReadMode),
		iterator.options.Source.FetchRows,
		iterator.options.MaxRecordBytes,
		afterKey,
		&iterator.telemetry,
	)
	if err != nil {
		return err
	}
	iterator.current = current
	if iterator.hasResume &&
		iterator.options.Source.ReadMode != config.PostgreSQLReadKeyset {
		for skipped := int64(0); skipped < iterator.resume.consumed; skipped++ {
			if err := ctx.Err(); err != nil {
				_ = iterator.closeCurrent()
				return err
			}
			if _, err := iterator.current.Next(ctx); err != nil {
				_ = iterator.closeCurrent()
				if errors.Is(err, io.EOF) {
					return errors.New(
						"PostgreSQL resume token consumed count exceeds mapping rows",
					)
				}
				return err
			}
		}
	}
	iterator.hasResume = false
	return nil
}

func (iterator *Iterator) closeCurrent() error {
	if iterator.current == nil {
		return nil
	}
	err := iterator.current.Close()
	iterator.current = nil
	return err
}

func (iterator *Iterator) handleMalformed(
	ctx context.Context,
	mapping compiledMapping,
	recordErr error,
) error {
	if iterator.options.OnMalformed == nil {
		return recordErr
	}
	iterator.rejected++
	if iterator.rejected > iterator.options.RejectLimit {
		return fmt.Errorf(
			"PostgreSQL reject limit %d exceeded: %w",
			iterator.options.RejectLimit,
			recordErr,
		)
	}
	position, err := iterator.buildPosition(mapping, iterator.rejected)
	if err != nil {
		return err
	}
	iterator.lastPosition = position
	if err := iterator.options.OnMalformed(ctx, MalformedRecord{
		Position: position, Err: recordErr,
	}); err != nil {
		return fmt.Errorf("write PostgreSQL quarantine record: %w", err)
	}
	return nil
}

func (iterator *Iterator) buildPosition(
	mapping compiledMapping,
	rejected int,
) (model.SourcePosition, error) {
	token, err := formatResumeToken(resumeState{
		fingerprint: iterator.fingerprint, mappingIndex: iterator.mappingIndex,
		mappingKind: mapping.kind, consumed: iterator.consumed,
		rejected: rejected, key: iterator.lastKey,
	})
	if err != nil {
		return model.SourcePosition{}, err
	}
	return model.SourcePosition{
		Connector: "postgresql", Resource: mapping.resource(),
		Line: iterator.consumed, Token: token,
	}, nil
}

func (iterator *Iterator) decodeRecord(
	ctx context.Context,
	mapping compiledMapping,
	raw []byte,
) (model.Record, int64, error) {
	if int64(len(raw)) > iterator.options.MaxRecordBytes {
		return model.Record{}, 0, fmt.Errorf(
			"PostgreSQL row exceeds maximum size of %d bytes",
			iterator.options.MaxRecordBytes,
		)
	}
	document, err := decodeObject(raw)
	if err != nil {
		return model.Record{}, 0, err
	}
	properties, encoded, propertySize, err := iterator.buildProperties(
		ctx, document, mapping.properties,
	)
	if err != nil {
		return model.Record{}, 0, err
	}
	if mapping.kind == vertexMapping {
		externalID, err := resolveExternalID(document, mapping.idField, "idField")
		if err != nil {
			return model.Record{}, 0, err
		}
		vertex := model.Vertex{
			Label: mapping.label, Namespace: mapping.namespace,
			ExternalID: externalID, Properties: properties,
			EncodedProperties: encoded,
		}
		size := saturatingAdd(
			vertexBaseSize,
			int64(len(vertex.Label)+len(vertex.Namespace)+len(vertex.ExternalID)),
		)
		return model.VertexRecord(vertex), saturatingAdd(size, propertySize), nil
	}

	var externalID model.ExternalID
	if mapping.externalIDField != "" {
		externalID, err = resolveExternalID(
			document, mapping.externalIDField, "externalIdField",
		)
		if err != nil {
			return model.Record{}, 0, err
		}
	}
	startID, err := resolveExternalID(document, mapping.start.Field, "start field")
	if err != nil {
		return model.Record{}, 0, err
	}
	endID, err := resolveExternalID(document, mapping.end.Field, "end field")
	if err != nil {
		return model.Record{}, 0, err
	}
	startNamespace := mapping.start.Namespace
	if startNamespace == "" {
		startNamespace = string(mapping.namespace)
	}
	endNamespace := mapping.end.Namespace
	if endNamespace == "" {
		endNamespace = string(mapping.namespace)
	}
	edge := model.Edge{
		Label: mapping.label, Namespace: mapping.namespace,
		ExternalID: externalID,
		Start: model.Endpoint{
			Label:     model.Label(mapping.start.Label),
			Namespace: model.Namespace(startNamespace), ExternalID: startID,
		},
		End: model.Endpoint{
			Label:     model.Label(mapping.end.Label),
			Namespace: model.Namespace(endNamespace), ExternalID: endID,
		},
		Properties: properties, EncodedProperties: encoded,
	}
	size := saturatingAdd(edgeBaseSize, int64(
		len(edge.Label)+len(edge.Namespace)+len(edge.ExternalID)+
			len(edge.Start.Label)+len(edge.Start.Namespace)+len(edge.Start.ExternalID)+
			len(edge.End.Label)+len(edge.End.Namespace)+len(edge.End.ExternalID),
	))
	return model.EdgeRecord(edge), saturatingAdd(size, propertySize), nil
}

func (iterator *Iterator) buildProperties(
	ctx context.Context,
	document map[string]any,
	properties []compiledProperty,
) (model.Properties, []byte, int64, error) {
	values := make(model.Properties, len(properties))
	var size int64
	for _, property := range properties {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, err
		}
		raw, ok := document[property.field]
		if !ok {
			return nil, nil, 0, fmt.Errorf(
				"PostgreSQL row is missing property %q", property.name,
			)
		}
		value, err := convertValue(raw, 0)
		if err != nil {
			return nil, nil, 0, fmt.Errorf(
				"PostgreSQL property %q: %w", property.name, err,
			)
		}
		values[property.name] = value
		size = saturatingAdd(
			size,
			saturatingAdd(int64(len(property.name)), estimateValueSize(value)),
		)
	}
	if !iterator.options.PreencodeProperties {
		return values, nil, size, nil
	}
	encoded, err := model.EncodeProperties(values)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("encode PostgreSQL properties: %w", err)
	}
	return nil, encoded, int64(len(encoded)), nil
}

func setPosition(record *model.Record, position model.SourcePosition) {
	switch record.Kind() {
	case model.RecordVertex:
		record.Vertex.Position = position
	case model.RecordEdge:
		record.Edge.Position = position
	}
}

func (iterator *Iterator) RejectionCheckpoint() (int64, model.SourcePosition) {
	iterator.mu.Lock()
	defer iterator.mu.Unlock()
	return int64(iterator.rejected), iterator.lastPosition
}

func (iterator *Iterator) Telemetry() Telemetry {
	return iterator.telemetry.snapshot()
}

func (iterator *Iterator) Close() error {
	iterator.mu.Lock()
	defer iterator.mu.Unlock()
	if iterator.closed {
		return iterator.closeErr
	}
	iterator.closed = true
	iterator.closeErr = errors.Join(iterator.closeErr, iterator.closeCurrent())
	_ = iterator.closeResources()
	return iterator.closeErr
}

func (iterator *Iterator) closeResources() error {
	iterator.closeOnce.Do(func() {
		iterator.closeErr = errors.Join(iterator.closeErr, iterator.coordinator.Close())
	})
	return iterator.closeErr
}

const (
	vertexBaseSize int64 = 384
	edgeBaseSize   int64 = 640
	propertyBase   int64 = 160
	mapBase        int64 = 64
	firstMapBucket int64 = 768
)

func estimateValueSize(value model.Value) int64 {
	switch value.Kind {
	case model.ValueString:
		return saturatingAdd(propertyBase, int64(len(value.String)))
	case model.ValueList:
		size := propertyBase
		for _, item := range value.List {
			size = saturatingAdd(size, estimateValueSize(item))
		}
		return size
	case model.ValueObject:
		size := saturatingAdd(propertyBase, mapBase)
		if len(value.Object) > 0 {
			size = saturatingAdd(size, firstMapBucket)
		}
		for name, item := range value.Object {
			size = saturatingAdd(
				size,
				saturatingAdd(int64(len(name)), estimateValueSize(item)),
			)
		}
		return size
	default:
		return propertyBase
	}
}

func saturatingAdd(left, right int64) int64 {
	if right > 0 && left > math.MaxInt64-right {
		return math.MaxInt64
	}
	return left + right
}

var (
	_ sourcecontract.Iterator              = (*Iterator)(nil)
	_ sourcecontract.RejectionCheckpointer = (*Iterator)(nil)
	_ sourcecontract.TelemetryProvider     = (*Iterator)(nil)
)
