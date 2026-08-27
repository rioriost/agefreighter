package cosmos

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

// defaultContinuationTokenLimitKB bounds the size of continuation tokens
// the service is asked to return, avoiding excessively large tokens that
// are expensive to store/transmit as checkpoints.
const defaultContinuationTokenLimitKB int32 = 8

// MalformedRecord describes a single Cosmos document that could not be
// mapped into a model.Record. It deliberately does not include the source
// document: only its position (which itself never contains document
// content or a full log of prior tokens beyond the opaque resume token).
type MalformedRecord struct {
	Position model.SourcePosition
	Err      error
}

// MalformedHandler is invoked for each malformed record when Iterator is
// configured with a positive reject limit. Returning a non-nil error stops
// iteration.
type MalformedHandler func(context.Context, MalformedRecord) error

// IteratorOptions configures a Cosmos DB for NoSQL Iterator.
type IteratorOptions struct {
	Namespace string
	Source    config.CosmosSource
	// Client is the injected query-client abstraction. Production callers
	// should construct one with NewSDKQueryClient and reuse it for the
	// lifetime of the Iterator.
	Client QueryClient
	// AfterToken, when non-empty, resumes iteration from a token
	// previously returned via model.SourcePosition.Token.
	AfterToken string
	// RejectLimit bounds how many malformed records may be quarantined
	// before iteration fails. OnMalformed is required whenever RejectLimit
	// is positive.
	RejectLimit    int
	MaxRecordBytes int64
	MaxProperties  int
	OnMalformed    MalformedHandler
	// PreencodeProperties builds the canonical AGE fast-path encoding
	// (model.EncodeProperties) instead of populating model.Properties.
	PreencodeProperties bool
}

// Iterator is a bounded source.Iterator over Cosmos DB for NoSQL vertex and
// edge queries. It preserves configured mapping order (every vertex mapping
// before every edge mapping), retains only the current page in memory, and
// reuses a single injected QueryClient across every mapping.
type Iterator struct {
	options      IteratorOptions
	mappings     []compiledMapping
	fingerprint  string
	mappingIndex int
	current      *openPage

	resume    resumeState
	hasResume bool

	rejected     int
	lastPosition model.SourcePosition

	telemetry telemetryState

	closed       bool
	lastCloseErr error
}

// openPage tracks the single retained page of raw documents for the
// mapping currently being iterated.
type openPage struct {
	items     [][]byte
	itemIndex int

	// openHasContinuation/openContinuation describe the continuation token
	// used to fetch (OPEN) this exact page; both are zero-valued for a
	// mapping's first page.
	openHasContinuation bool
	openContinuation    string

	// hasNextContinuation/nextContinuation describe the continuation
	// returned by this page, used to open the next one.
	hasNextContinuation bool
	nextContinuation    string

	pageBytes int64
}

// NewIterator compiles the source's mappings, resolves any resume token,
// and returns a ready-to-use Iterator. It performs no I/O against Cosmos
// itself; the first query pager is opened on the first call to Next.
func NewIterator(ctx context.Context, options IteratorOptions) (*Iterator, error) {
	if ctx == nil {
		return nil, errors.New("Cosmos iterator context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.Namespace == "" {
		return nil, errors.New("Cosmos namespace is required")
	}
	if options.Client == nil {
		return nil, errors.New("Cosmos query client is required")
	}
	if options.RejectLimit < 0 {
		return nil, errors.New("Cosmos reject limit cannot be negative")
	}
	if options.OnMalformed == nil && options.RejectLimit > 0 {
		return nil, errors.New("Cosmos malformed handler is required when reject limit is positive")
	}
	if options.MaxRecordBytes == 0 {
		options.MaxRecordBytes = 16 << 20
	}
	if options.MaxRecordBytes < 1 {
		return nil, errors.New("Cosmos maximum record bytes must be positive")
	}
	if options.MaxProperties == 0 {
		options.MaxProperties = 1024
	}
	if options.MaxProperties < 1 {
		return nil, errors.New("Cosmos maximum properties must be positive")
	}
	if options.Source.PageSize < 1 || options.Source.PageSize > 1000 {
		return nil, errors.New("Cosmos page size must be between 1 and 1000")
	}

	mappings, err := buildMappings(ctx, options.Namespace, options.Source, options.MaxProperties)
	if err != nil {
		return nil, err
	}
	fingerprint, err := bindFingerprint(
		options.Source.Endpoint,
		options.Source.Database,
		options.Namespace,
		int32(options.Source.PageSize),
		mappings,
	)
	if err != nil {
		return nil, err
	}
	iterator := &Iterator{options: options, mappings: mappings, fingerprint: fingerprint}
	if options.AfterToken != "" {
		resume, err := parseResumeToken(options.AfterToken)
		if err != nil {
			return nil, err
		}
		if resume.fingerprint != fingerprint {
			return nil, errors.New("Cosmos source fingerprint changed since checkpoint")
		}
		if resume.mappingIndex >= len(mappings) {
			return nil, errors.New("Cosmos resume token mapping index is out of range")
		}
		if resume.mappingKind != mappings[resume.mappingIndex].kind {
			return nil, errors.New("Cosmos resume token mapping kind does not match the configured mapping")
		}
		if resume.rejected > options.RejectLimit {
			return nil, errors.New("Cosmos resume token exceeds configured reject limit")
		}
		iterator.resume = resume
		iterator.hasResume = true
		iterator.mappingIndex = resume.mappingIndex
		iterator.rejected = resume.rejected
		iterator.lastPosition.Token = options.AfterToken
	}
	return iterator, nil
}

// Next implements source.Iterator. It returns io.EOF once every mapping's
// documents have been exhausted.
func (iterator *Iterator) Next(ctx context.Context) (sourcecontract.Item, error) {
	if iterator.closed {
		return sourcecontract.Item{}, errors.New("Cosmos iterator is closed")
	}
	for {
		if err := ctx.Err(); err != nil {
			return sourcecontract.Item{}, err
		}
		if iterator.current == nil {
			if iterator.mappingIndex >= len(iterator.mappings) {
				return sourcecontract.Item{}, io.EOF
			}
			if err := iterator.openNextPage(ctx); err != nil {
				return sourcecontract.Item{}, err
			}
			continue
		}

		page := iterator.current
		if page.itemIndex >= len(page.items) {
			if !page.hasNextContinuation {
				iterator.current = nil
				iterator.mappingIndex++
				continue
			}
			if err := iterator.advancePage(ctx, page.nextContinuation); err != nil {
				return sourcecontract.Item{}, err
			}
			continue
		}

		mapping := iterator.mappings[iterator.mappingIndex]
		raw := page.items[page.itemIndex]
		page.itemIndex++

		record, size, err := iterator.decodeRecord(ctx, mapping, raw)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return sourcecontract.Item{}, ctxErr
			}
			if handledErr := iterator.handleMalformed(ctx, page, mapping, err); handledErr != nil {
				return sourcecontract.Item{}, handledErr
			}
			continue
		}
		position := iterator.buildPosition(page, mapping, iterator.rejected)
		setPosition(&record, position)
		iterator.lastPosition = position
		return sourcecontract.Item{
			Record:      record,
			SampleBytes: size,
			SizeBytes:   saturatingAdd(size, page.pageBytes),
		}, nil
	}
}

// Close implements source.Iterator. It is idempotent and prompt: it holds
// no file handles or long-lived network connections of its own (those
// belong to the injected QueryClient, which the caller owns).
func (iterator *Iterator) Close() error {
	if iterator.closed {
		return iterator.lastCloseErr
	}
	iterator.closed = true
	iterator.current = nil
	if closer, ok := iterator.options.Client.(Closer); ok {
		iterator.lastCloseErr = closer.Close()
	}
	return iterator.lastCloseErr
}

// RejectionCheckpoint reports how many malformed records have been
// quarantined so far and the position of the most recently handled record
// (successful or quarantined), mirroring the CSV source's contract.
func (iterator *Iterator) RejectionCheckpoint() (int64, model.SourcePosition) {
	return int64(iterator.rejected), iterator.lastPosition
}

// Telemetry returns a point-in-time, non-secret snapshot of cumulative
// diagnostics. It never exposes full continuation tokens or document
// content.
func (iterator *Iterator) Telemetry() Telemetry {
	var throttled int64
	if observer, ok := iterator.options.Client.(ThrottleObserver); ok {
		throttled = observer.ThrottledRequests()
	}
	return iterator.telemetry.snapshot(throttled)
}

// openNextPage opens the first page of the mapping at iterator.mappingIndex,
// honoring any pending resume state for that exact mapping.
func (iterator *Iterator) openNextPage(ctx context.Context) error {
	mapping := iterator.mappings[iterator.mappingIndex]

	resuming := iterator.hasResume && iterator.mappingIndex == iterator.resume.mappingIndex
	hasContinuation := false
	continuation := ""
	skip := 0
	if resuming {
		hasContinuation = iterator.resume.hasContinuation
		continuation = iterator.resume.continuation
		skip = iterator.resume.consumed
	}

	page, err := iterator.fetchPage(ctx, mapping, hasContinuation, continuation)
	if err != nil {
		return err
	}
	if resuming {
		if skip > len(page.Items) {
			return errors.New("Cosmos resume token skip count exceeds the re-fetched page")
		}
		iterator.hasResume = false
	}
	iterator.current = &openPage{
		items:               page.Items,
		itemIndex:           skip,
		openHasContinuation: hasContinuation,
		openContinuation:    continuation,
		hasNextContinuation: page.HasContinuation,
		nextContinuation:    page.ContinuationToken,
		pageBytes:           computePageBytes(page.Items),
	}
	iterator.telemetry.recordPage(page)
	return nil
}

// advancePage fetches the next page of the current mapping, opened by the
// continuation token returned by the previously exhausted page.
func (iterator *Iterator) advancePage(ctx context.Context, continuation string) error {
	mapping := iterator.mappings[iterator.mappingIndex]
	page, err := iterator.fetchPage(ctx, mapping, true, continuation)
	if err != nil {
		return err
	}
	iterator.current = &openPage{
		items:               page.Items,
		itemIndex:           0,
		openHasContinuation: true,
		openContinuation:    continuation,
		hasNextContinuation: page.HasContinuation,
		nextContinuation:    page.ContinuationToken,
		pageBytes:           computePageBytes(page.Items),
	}
	iterator.telemetry.recordPage(page)
	return nil
}

func (iterator *Iterator) fetchPage(
	ctx context.Context,
	mapping compiledMapping,
	hasContinuation bool,
	continuation string,
) (Page, error) {
	if err := ctx.Err(); err != nil {
		return Page{}, err
	}
	pager, err := iterator.options.Client.NewQueryPager(
		mapping.container,
		mapping.query,
		mapping.parameters,
		QueryOptions{
			PageSizeHint:             int32(iterator.options.Source.PageSize),
			ContinuationToken:        continuation,
			HasContinuationToken:     hasContinuation,
			ContinuationTokenLimitKB: defaultContinuationTokenLimitKB,
		},
	)
	if err != nil {
		return Page{}, fmt.Errorf("open Cosmos query pager: %w", err)
	}
	page, err := pager.NextPage(ctx)
	if err != nil {
		return Page{}, fmt.Errorf("fetch Cosmos query page: %w", err)
	}
	return page, nil
}

// buildPosition renders the SourcePosition attached to a record (or
// malformed record) currently at the given page/mapping, with rejected
// reflecting the count to embed in the resume token.
func (iterator *Iterator) buildPosition(
	page *openPage,
	mapping compiledMapping,
	rejected int,
) model.SourcePosition {
	token := formatResumeToken(resumeState{
		fingerprint:     iterator.fingerprint,
		mappingIndex:    iterator.mappingIndex,
		mappingKind:     mapping.kind,
		hasContinuation: page.openHasContinuation,
		continuation:    page.openContinuation,
		consumed:        page.itemIndex,
		rejected:        rejected,
	})
	return model.SourcePosition{
		Connector: string(config.SourceCosmos),
		Resource: fmt.Sprintf(
			"%s/%s[%s]", iterator.options.Source.Database, mapping.container, mapping.label,
		),
		Token: token,
	}
}

func (iterator *Iterator) handleMalformed(
	ctx context.Context,
	page *openPage,
	mapping compiledMapping,
	recordErr error,
) error {
	if iterator.options.OnMalformed == nil {
		return recordErr
	}
	iterator.rejected++
	if iterator.rejected > iterator.options.RejectLimit {
		return fmt.Errorf(
			"Cosmos reject limit %d exceeded: %w", iterator.options.RejectLimit, recordErr,
		)
	}
	position := iterator.buildPosition(page, mapping, iterator.rejected)
	iterator.lastPosition = position
	if err := iterator.options.OnMalformed(ctx, MalformedRecord{
		Position: position,
		Err:      recordErr,
	}); err != nil {
		return fmt.Errorf("write Cosmos quarantine record: %w", err)
	}
	return nil
}

// decodeRecord decodes and maps a single raw Cosmos document into a
// model.Record, returning a conservative estimate of the memory the mapped
// record retains.
func (iterator *Iterator) decodeRecord(
	ctx context.Context,
	mapping compiledMapping,
	raw []byte,
) (model.Record, int64, error) {
	if int64(len(raw)) > iterator.options.MaxRecordBytes {
		return model.Record{}, 0, fmt.Errorf(
			"Cosmos document exceeds maximum size of %d bytes", iterator.options.MaxRecordBytes,
		)
	}
	document, err := decodeDocument(raw)
	if err != nil {
		return model.Record{}, 0, err
	}
	if mapping.documentFormat == config.CosmosDocumentGremlin {
		return iterator.decodeGremlinRecord(ctx, mapping, document)
	}

	if mapping.kind == vertexMapping {
		externalID, err := resolveRequiredString(document, mapping.idField, "idField")
		if err != nil {
			return model.Record{}, 0, err
		}
		properties, encoded, propertiesSize, err := iterator.buildProperties(ctx, document, mapping.properties)
		if err != nil {
			return model.Record{}, 0, err
		}
		vertex := model.Vertex{
			Label:             mapping.label,
			Namespace:         mapping.namespace,
			ExternalID:        model.ExternalID(externalID),
			Properties:        properties,
			EncodedProperties: encoded,
		}
		size := saturatingAdd(propertiesSize, vertexBaseSize)
		size = saturatingAdd(size, int64(len(vertex.Label)+len(vertex.Namespace)+len(vertex.ExternalID)))
		return model.VertexRecord(vertex), size, nil
	}

	var externalID model.ExternalID
	if mapping.hasExternalID {
		value, err := resolveRequiredString(document, mapping.externalIDField, "externalIdField")
		if err != nil {
			return model.Record{}, 0, err
		}
		externalID = model.ExternalID(value)
	}
	startID, err := resolveRequiredString(document, mapping.startField, "start field")
	if err != nil {
		return model.Record{}, 0, err
	}
	endID, err := resolveRequiredString(document, mapping.endField, "end field")
	if err != nil {
		return model.Record{}, 0, err
	}
	properties, encoded, propertiesSize, err := iterator.buildProperties(ctx, document, mapping.properties)
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
		Label:      mapping.label,
		Namespace:  mapping.namespace,
		ExternalID: externalID,
		Start: model.Endpoint{
			Label:      model.Label(mapping.start.Label),
			Namespace:  model.Namespace(startNamespace),
			ExternalID: model.ExternalID(startID),
		},
		End: model.Endpoint{
			Label:      model.Label(mapping.end.Label),
			Namespace:  model.Namespace(endNamespace),
			ExternalID: model.ExternalID(endID),
		},
		Properties:        properties,
		EncodedProperties: encoded,
	}
	size := saturatingAdd(propertiesSize, edgeBaseSize)
	size = saturatingAdd(size, int64(
		len(edge.Label)+len(edge.Namespace)+len(edge.ExternalID)+
			len(edge.Start.Label)+len(edge.Start.Namespace)+len(edge.Start.ExternalID)+
			len(edge.End.Label)+len(edge.End.Namespace)+len(edge.End.ExternalID),
	))
	return model.EdgeRecord(edge), size, nil
}

func (iterator *Iterator) buildProperties(
	ctx context.Context,
	document any,
	properties []compiledProperty,
) (model.Properties, []byte, int64, error) {
	values := make(model.Properties, len(properties))
	var size int64
	for _, property := range properties {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, err
		}
		raw, ok := property.pointer.resolve(document)
		if !ok {
			return nil, nil, 0, fmt.Errorf("Cosmos document is missing property %q", property.name)
		}
		value, err := convertValue(raw, 0)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("Cosmos property %q: %w", property.name, err)
		}
		values[property.name] = value
		size = saturatingAdd(size, saturatingAdd(estimateValueSize(value), int64(len(property.name))))
	}
	if !iterator.options.PreencodeProperties {
		return values, nil, size, nil
	}
	encoded, err := model.EncodeProperties(values)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("encode Cosmos properties: %w", err)
	}
	return nil, encoded, int64(len(encoded)), nil
}

// setPosition attaches position to the vertex or edge inside record.
func setPosition(record *model.Record, position model.SourcePosition) {
	switch record.Kind() {
	case model.RecordVertex:
		record.Vertex.Position = position
	case model.RecordEdge:
		record.Edge.Position = position
	}
}

const (
	vertexBaseSize int64 = 384
	edgeBaseSize   int64 = 640
	propertyBase   int64 = 160
	mapBase        int64 = 64
	firstMapBucket int64 = 768
)

// estimateValueSize conservatively estimates the retained heap size of a
// model.Value, recursing into lists and objects.
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
		if len(value.Object) != 0 {
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

// computePageBytes conservatively estimates the retained memory of the
// current page's raw items (their raw JSON bytes plus per-item overhead).
// It is charged, in full, to every record produced from that page: an
// intentional over-count so pipeline admission control stays safe even
// though only one page is ever retained at a time.
func computePageBytes(items [][]byte) int64 {
	const perItemOverhead int64 = 48
	var total int64
	for _, item := range items {
		total = saturatingAdd(total, saturatingAdd(int64(len(item)), perItemOverhead))
	}
	return total
}

// saturatingAdd adds a and b, clamping to math.MaxInt64 instead of
// overflowing/wrapping.
func saturatingAdd(a, b int64) int64 {
	if a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}

var _ sourcecontract.Iterator = (*Iterator)(nil)
