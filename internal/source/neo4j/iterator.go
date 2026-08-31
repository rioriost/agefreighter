package neo4j

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

type MalformedRecord struct {
	Position model.SourcePosition
	Err      error
}

type MalformedHandler func(context.Context, MalformedRecord) error

type IteratorOptions struct {
	Namespace           string
	Source              config.Neo4jSource
	Client              Client
	AfterToken          string
	RejectLimit         int
	MaxRecordBytes      int64
	MaxProperties       int
	OnMalformed         MalformedHandler
	PreencodeProperties bool
	ProfileBudget       *sourcecontract.ProfileBudget
}

type Iterator struct {
	mu          sync.Mutex
	options     IteratorOptions
	mappings    []compiledMapping
	fingerprint string
	current     RecordStream
	lifetime    context.Context
	cancel      context.CancelFunc

	mappingIndex int
	consumed     int64
	lastKey      *int64
	pageRows     int
	rejected     int
	lastPosition model.SourcePosition

	telemetry   telemetryState
	closed      bool
	exhausted   bool
	terminalErr error
	closeOnce   sync.Once
	closeErr    error
}

func NewIterator(ctx context.Context, options IteratorOptions) (*Iterator, error) {
	if ctx == nil {
		return nil, errors.New("Neo4j iterator context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.Client == nil {
		return nil, errors.New("Neo4j client is required")
	}
	if options.Source.SourceID == "" {
		return nil, errors.New("Neo4j sourceId is required")
	}
	if options.Source.URI == "" {
		return nil, errors.New("Neo4j URI is required")
	}
	if options.Source.Database == "" {
		return nil, errors.New("Neo4j database is required")
	}
	if options.Source.FetchRows < 1 || options.Source.FetchRows > 100_000 {
		return nil, errors.New("Neo4j fetch rows must be between 1 and 100000")
	}
	switch options.Source.MultiLabelPolicy {
	case config.Neo4jMultiLabelConfigured, config.Neo4jMultiLabelReject:
	default:
		return nil, errors.New("Neo4j multi-label policy must be configured or reject")
	}
	if options.RejectLimit < 0 {
		return nil, errors.New("Neo4j reject limit cannot be negative")
	}
	if options.RejectLimit > 0 && options.OnMalformed == nil {
		return nil, errors.New("Neo4j malformed handler is required when reject limit is positive")
	}
	if options.MaxRecordBytes == 0 {
		options.MaxRecordBytes = 16 << 20
	}
	if options.MaxRecordBytes < 1 {
		return nil, errors.New("Neo4j maximum record bytes must be positive")
	}
	if options.MaxProperties == 0 {
		options.MaxProperties = 1024
	}
	if options.MaxProperties < 1 {
		return nil, errors.New("Neo4j maximum properties must be positive")
	}
	mappings, err := buildMappings(
		ctx, options.Namespace, options.Source, options.MaxProperties,
	)
	if err != nil {
		return nil, err
	}
	fingerprint, err := bindFingerprint(options.Source, options.Namespace, mappings)
	if err != nil {
		return nil, err
	}
	lifetime, cancel := context.WithCancel(context.Background())
	iterator := &Iterator{
		options: options, mappings: mappings, fingerprint: fingerprint,
		lifetime: lifetime, cancel: cancel,
	}
	if options.AfterToken != "" {
		resume, err := parseResumeToken(options.AfterToken)
		if err != nil {
			return nil, err
		}
		if resume.fingerprint != fingerprint {
			return nil, errors.New("Neo4j source fingerprint changed since checkpoint")
		}
		if resume.mappingIndex >= len(mappings) {
			return nil, errors.New("Neo4j resume token mapping index is out of range")
		}
		if resume.mappingKind != mappings[resume.mappingIndex].kind {
			return nil, errors.New("Neo4j resume token mapping kind does not match configuration")
		}
		if resume.rejected > options.RejectLimit {
			return nil, errors.New("Neo4j resume token exceeds configured reject limit")
		}
		iterator.mappingIndex = resume.mappingIndex
		iterator.consumed = resume.consumed
		iterator.lastKey = cloneKey(resume.lastKey)
		iterator.rejected = resume.rejected
		iterator.lastPosition.Token = options.AfterToken
	}
	return iterator, nil
}

func (iterator *Iterator) Next(ctx context.Context) (sourcecontract.Item, error) {
	if ctx == nil {
		return sourcecontract.Item{}, errors.New("Neo4j iterator context is required")
	}
	opCtx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(iterator.lifetime, cancel)
	defer func() {
		stop()
		cancel()
	}()
	ctx = opCtx
	iterator.mu.Lock()
	defer iterator.mu.Unlock()
	if iterator.closed {
		return sourcecontract.Item{}, errors.New("Neo4j iterator is closed")
	}
	if iterator.terminalErr != nil {
		return sourcecontract.Item{}, iterator.terminalErr
	}
	if iterator.exhausted {
		return sourcecontract.Item{}, io.EOF
	}
	for {
		if err := ctx.Err(); err != nil {
			return sourcecontract.Item{}, err
		}
		if err := iterator.options.ProfileBudget.CanProcess(); err != nil {
			return sourcecontract.Item{}, iterator.fail(err)
		}
		if iterator.current == nil {
			if iterator.mappingIndex >= len(iterator.mappings) {
				iterator.exhausted = true
				return sourcecontract.Item{}, io.EOF
			}
			if err := iterator.openCurrent(ctx); err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					return sourcecontract.Item{}, ctxErr
				}
				return sourcecontract.Item{}, iterator.fail(err)
			}
		}
		record, err := iterator.current.Next(ctx)
		if errors.Is(err, io.EOF) {
			mapping := iterator.mappings[iterator.mappingIndex]
			pageFull := mapping.paged &&
				iterator.pageRows == iterator.options.Source.FetchRows
			if closeErr := iterator.closeCurrent(ctx); closeErr != nil {
				iterator.closeErr = errors.Join(iterator.closeErr, closeErr)
				iterator.mappingIndex++
				iterator.consumed = 0
				iterator.lastKey = nil
				return sourcecontract.Item{}, iterator.fail(closeErr)
			}
			if pageFull {
				iterator.pageRows = 0
				continue
			}
			iterator.mappingIndex++
			iterator.consumed = 0
			iterator.lastKey = nil
			iterator.pageRows = 0
			continue
		}
		if err != nil {
			iterator.telemetry.failure()
			if ctxErr := ctx.Err(); ctxErr != nil {
				return sourcecontract.Item{}, ctxErr
			}
			return sourcecontract.Item{}, iterator.fail(
				safeError(ctx, "read Neo4j query result", err),
			)
		}
		iterator.pageRows++
		rawSize, err := estimateRecordSize(record, math.MaxInt64)
		iterator.telemetry.record(rawSize)
		if budgetErr := iterator.options.ProfileBudget.Charge(
			sourcecontract.ProfileBudgetUsage{
				Rows: 1, DecodedInputBytes: rawSize,
			},
		); budgetErr != nil {
			return sourcecontract.Item{}, iterator.fail(budgetErr)
		}
		mapping := iterator.mappings[iterator.mappingIndex]
		key, keyErr := extractKey(record, mapping.keyField)
		if keyErr != nil {
			return sourcecontract.Item{}, iterator.fail(keyErr)
		}
		if iterator.lastKey != nil && key <= *iterator.lastKey {
			return sourcecontract.Item{}, iterator.fail(errors.New(
				"Neo4j keyField values must be unique and strictly increasing",
			))
		}
		iterator.lastKey = &key
		iterator.consumed++

		if err != nil {
			if handled := iterator.handleMalformed(ctx, mapping, err); handled != nil {
				return sourcecontract.Item{}, iterator.fail(handled)
			}
			continue
		}
		if rawSize > iterator.options.MaxRecordBytes {
			err := fmt.Errorf(
				"Neo4j record exceeds maximum size of %d bytes",
				iterator.options.MaxRecordBytes,
			)
			if handled := iterator.handleMalformed(ctx, mapping, err); handled != nil {
				return sourcecontract.Item{}, iterator.fail(handled)
			}
			continue
		}
		mapped, size, err := iterator.decodeRecord(ctx, mapping, record)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return sourcecontract.Item{}, iterator.fail(ctxErr)
			}
			if handled := iterator.handleMalformed(ctx, mapping, err); handled != nil {
				return sourcecontract.Item{}, iterator.fail(handled)
			}
			continue
		}
		position, err := iterator.buildPosition(mapping, iterator.rejected)
		if err != nil {
			return sourcecontract.Item{}, iterator.fail(err)
		}

		setPosition(&mapped, position)
		iterator.lastPosition = position
		return sourcecontract.Item{
			Record: mapped, SizeBytes: saturatingAdd(size, rawSize),
		}, nil
	}
}

func (iterator *Iterator) fail(err error) error {
	iterator.terminalErr = err
	return err
}

func (iterator *Iterator) openCurrent(ctx context.Context) error {
	if err := iterator.options.ProfileBudget.Full(); err != nil {
		return err
	}
	mapping := iterator.mappings[iterator.mappingIndex]
	var afterKey any
	if iterator.lastKey != nil {
		afterKey = *iterator.lastKey
	}
	parameters := map[string]any{"afterKey": afterKey}
	query := mapping.query
	if iterator.lastKey == nil && mapping.initialQuery != "" {
		query = mapping.initialQuery
		delete(parameters, "afterKey")
	}
	if mapping.paged {
		parameters["pageRows"] = iterator.options.Source.FetchRows
	}
	stream, err := iterator.options.Client.Query(
		ctx, query, parameters,
	)
	if err != nil {
		iterator.telemetry.failure()
		return safeError(ctx, "open Neo4j query", err)
	}
	iterator.telemetry.query()
	if err := iterator.options.ProfileBudget.Charge(
		sourcecontract.ProfileBudgetUsage{Pages: 1},
	); err != nil {
		return errors.Join(err, stream.Close(ctx))
	}
	iterator.current = stream
	iterator.pageRows = 0
	return nil
}

func (iterator *Iterator) closeCurrent(ctx context.Context) error {
	if iterator.current == nil {
		return nil
	}
	err := iterator.current.Close(ctx)
	iterator.current = nil
	if err != nil {
		iterator.telemetry.failure()
		return safeError(ctx, "close Neo4j query", err)
	}
	return nil
}

func (iterator *Iterator) decodeRecord(
	ctx context.Context,
	mapping compiledMapping,
	record Record,
) (model.Record, int64, error) {
	if mapping.kind == vertexMapping {
		externalID, err := resolveExternalID(record, mapping.idField, "idField")
		if err != nil {
			return model.Record{}, 0, err
		}
		properties, encoded, propertySize, err := iterator.buildProperties(
			ctx, record, mapping.properties,
		)
		if err != nil {
			return model.Record{}, 0, err
		}
		vertex := model.Vertex{
			Label: mapping.label, Namespace: mapping.namespace,
			ExternalID: externalID, Properties: properties, EncodedProperties: encoded,
		}
		size := saturatingAdd(vertexBaseSize, int64(
			len(vertex.Label)+len(vertex.Namespace)+len(vertex.ExternalID),
		))
		return model.VertexRecord(vertex), saturatingAdd(size, propertySize), nil
	}

	var externalID model.ExternalID
	var err error
	if mapping.externalIDField != "" {
		externalID, err = resolveExternalID(record, mapping.externalIDField, "externalIdField")
		if err != nil {
			return model.Record{}, 0, err
		}
	}
	startID, err := resolveExternalID(record, mapping.start.Field, "start field")
	if err != nil {
		return model.Record{}, 0, err
	}
	endID, err := resolveExternalID(record, mapping.end.Field, "end field")
	if err != nil {
		return model.Record{}, 0, err
	}
	properties, encoded, propertySize, err := iterator.buildProperties(
		ctx, record, mapping.properties,
	)
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
		Label: mapping.label, Namespace: mapping.namespace, ExternalID: externalID,
		Start: model.Endpoint{
			Label: model.Label(mapping.start.Label), Namespace: model.Namespace(startNamespace),
			ExternalID: startID,
		},
		End: model.Endpoint{
			Label: model.Label(mapping.end.Label), Namespace: model.Namespace(endNamespace),
			ExternalID: endID,
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
	record Record,
	properties []compiledProperty,
) (model.Properties, []byte, int64, error) {
	values := make(model.Properties, len(properties))
	var size int64
	for _, property := range properties {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, err
		}
		raw, ok := record.Get(property.field)
		if !ok {
			return nil, nil, 0, fmt.Errorf(
				"Neo4j record is missing property %q", property.name,
			)
		}
		value, err := convertValue(raw, 0, iterator.options.Source.MultiLabelPolicy)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("Neo4j property %q: %w", property.name, err)
		}
		values[property.name] = value
		size = saturatingAdd(
			size, saturatingAdd(int64(len(property.name)), estimateValueSize(value)),
		)
	}
	if !iterator.options.PreencodeProperties {
		return values, nil, size, nil
	}
	encoded, err := model.EncodeProperties(values)
	if err != nil {
		return nil, nil, 0, errors.New("encode Neo4j properties")
	}
	return nil, encoded, int64(len(encoded)), nil
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
		return fmt.Errorf("Neo4j reject limit %d exceeded: %w", iterator.options.RejectLimit, recordErr)
	}
	position, err := iterator.buildPosition(mapping, iterator.rejected)
	if err != nil {
		return err
	}
	iterator.lastPosition = position
	if err := iterator.options.OnMalformed(ctx, MalformedRecord{
		Position: position, Err: recordErr,
	}); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return errors.New("write Neo4j quarantine record failed")
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
		rejected: rejected, lastKey: cloneKey(iterator.lastKey),
	})
	if err != nil {
		return model.SourcePosition{}, err
	}
	return model.SourcePosition{
		Connector: "neo4j", Resource: mapping.resource(),
		Line: iterator.consumed, Token: token,
	}, nil
}

func cloneKey(key *int64) *int64 {
	if key == nil {
		return nil
	}
	copy := *key
	return &copy
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

func (iterator *Iterator) DetailedTelemetry() DetailedTelemetry {
	return iterator.telemetry.detailed()
}

func (iterator *Iterator) Close() error {
	iterator.cancel()
	iterator.mu.Lock()
	defer iterator.mu.Unlock()
	if iterator.closed {
		return iterator.closeErr
	}
	iterator.closed = true
	iterator.closeErr = errors.Join(
		iterator.closeErr, iterator.closeCurrent(context.Background()),
	)
	iterator.closeOnce.Do(func() {
		if err := iterator.options.Client.Close(); err != nil {
			iterator.closeErr = errors.Join(
				iterator.closeErr, safeError(nil, "close Neo4j client", err),
			)
		}
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
			size = saturatingAdd(size, saturatingAdd(int64(len(name)), estimateValueSize(item)))
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
