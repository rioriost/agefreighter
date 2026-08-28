package csv

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/hex"
	"encoding/json"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

type MalformedRecord struct {
	Position model.SourcePosition
	Fields   []string
	Err      error
}

type MalformedHandler func(context.Context, MalformedRecord) error

type IteratorOptions struct {
	Namespace           string
	Source              config.CSVSource
	AfterToken          string
	RejectLimit         int
	MaxRecordBytes      int64
	MaxFields           int
	MaxProperties       int
	OnMalformed         MalformedHandler
	PreencodeProperties bool
	OptimizeRFC4180     bool
	ProfileBudget       *sourcecontract.ProfileBudget
}

type Iterator struct {
	options      IteratorOptions
	mappings     []fileMapping
	mappingIndex int
	current      *openMapping
	resume       resumeToken
	hasResume    bool
	manifest     string
	manifestSet  bool
	rejected     int
	lastPosition model.SourcePosition
	closed       bool
	lastCloseErr error
	telemetry    telemetryState
}

type mappingKind uint8

const (
	vertexMapping mappingKind = iota + 1
	edgeMapping
)

type fileMapping struct {
	kind             mappingKind
	path             string
	label            model.Label
	namespace        model.Namespace
	idColumn         string
	externalIDColumn string
	start            config.EndpointMapping
	end              config.EndpointMapping
	properties       map[string]string
	format           config.DelimitedOptions
	fingerprint      string
	fingerprintInput []byte
}

type fingerprintMapping struct {
	Kind             mappingKind             `json:"kind"`
	Path             string                  `json:"path"`
	Label            model.Label             `json:"label"`
	Namespace        model.Namespace         `json:"namespace"`
	IDColumn         string                  `json:"idColumn,omitempty"`
	ExternalIDColumn string                  `json:"externalIdColumn,omitempty"`
	Start            config.EndpointMapping  `json:"start,omitempty"`
	End              config.EndpointMapping  `json:"end,omitempty"`
	Properties       map[string]string       `json:"properties,omitempty"`
	Format           config.DelimitedOptions `json:"format"`
}

type compiledProperty struct {
	name        string
	encodedName []byte
	index       int
}

type compiledMapping struct {
	id          int
	externalID  int
	start       int
	end         int
	properties  []compiledProperty
	fieldCount  int
	exactFields bool
}

type openMapping struct {
	file        *os.File
	gzip        *gzip.Reader
	parser      *Parser
	mapping     fileMapping
	compiled    compiledMapping
	recordIndex int64
}

type resumeToken struct {
	Mapping     int
	Record      int64
	Rejected    int
	Fingerprint string
}

func NewIterator(ctx context.Context, options IteratorOptions) (*Iterator, error) {
	if ctx == nil {
		return nil, errors.New("CSV iterator context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(options.Namespace) == "" {
		return nil, errors.New("CSV namespace is required")
	}
	if options.RejectLimit < 0 {
		return nil, errors.New("CSV reject limit cannot be negative")
	}
	if options.OnMalformed == nil && options.RejectLimit > 0 {
		return nil, errors.New("CSV malformed handler is required when reject limit is positive")
	}
	if options.MaxRecordBytes == 0 {
		options.MaxRecordBytes = 16 << 20
	}
	if options.MaxRecordBytes < 1 {
		return nil, errors.New("CSV maximum record bytes must be positive")
	}
	if options.MaxFields == 0 {
		options.MaxFields = 4096
	}
	if options.MaxFields < 1 {
		return nil, errors.New("CSV maximum fields must be positive")
	}
	if options.MaxProperties == 0 {
		options.MaxProperties = 1024
	}
	if options.MaxProperties < 1 {
		return nil, errors.New("CSV maximum properties must be positive")
	}

	mappings, err := buildMappings(ctx, options)
	if err != nil {
		return nil, err
	}
	iterator := &Iterator{options: options, mappings: mappings}
	if options.AfterToken != "" {
		iterator.resume, err = parseResumeToken(options.AfterToken)
		if err != nil {
			return nil, err
		}
		if iterator.resume.Mapping < 0 || iterator.resume.Mapping >= len(mappings) {
			return nil, errors.New("CSV resume token mapping is out of range")
		}
		iterator.mappingIndex = iterator.resume.Mapping
		iterator.hasResume = true
		iterator.lastPosition.Token = options.AfterToken
	}
	return iterator, nil
}

func (iterator *Iterator) Next(ctx context.Context) (sourcecontract.Item, error) {
	if iterator.closed {
		return sourcecontract.Item{}, errors.New("CSV iterator is closed")
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
				return sourcecontract.Item{}, io.EOF
			}
			if err := iterator.openCurrent(ctx); err != nil {
				return sourcecontract.Item{}, err
			}
		}

		fields, position, err := iterator.current.parser.ReadRecord(ctx)
		if errors.Is(err, io.EOF) {
			var verifyErr error
			if iterator.options.ProfileBudget == nil {
				verifyErr = iterator.current.verifyFingerprint(ctx)
			}
			closeErr := iterator.closeCurrent()
			if err := errors.Join(verifyErr, closeErr); err != nil {
				return sourcecontract.Item{}, err
			}
			iterator.mappingIndex++
			continue
		}
		if err != nil {
			if !errors.Is(err, sourcecontract.ErrProfileBudget) &&
				!errors.Is(err, context.Canceled) &&
				!errors.Is(err, context.DeadlineExceeded) {
				_ = iterator.options.ProfileBudget.Charge(
					sourcecontract.ProfileBudgetUsage{Rows: 1},
				)
			}
			return sourcecontract.Item{}, fmt.Errorf(
				"read CSV record at %s:%d: %w",
				positionFromError(err, position).Resource,
				positionFromError(err, position).Line,
				err,
			)
		}
		if err := iterator.options.ProfileBudget.Charge(
			sourcecontract.ProfileBudgetUsage{Rows: 1},
		); err != nil {
			return sourcecontract.Item{}, err
		}
		iterator.current.recordIndex++
		position.Token = formatResumeToken(resumeToken{
			Mapping:     iterator.mappingIndex,
			Record:      iterator.current.recordIndex,
			Rejected:    iterator.rejected,
			Fingerprint: iterator.manifest,
		})
		fieldCountInvalid := iterator.current.compiled.exactFields &&
			len(fields) != iterator.current.compiled.fieldCount
		if !iterator.current.compiled.exactFields {
			fieldCountInvalid = len(fields) < iterator.current.compiled.fieldCount
		}

		if fieldCountInvalid {
			if handledErr := iterator.handleMalformed(ctx, MalformedRecord{
				Position: position,
				Fields:   slices.Clone(fields),
				Err: fmt.Errorf(
					"CSV record has %d fields, expected %d",
					len(fields),
					iterator.current.compiled.fieldCount,
				),
			}); handledErr != nil {
				return sourcecontract.Item{}, handledErr
			}
			continue
		}

		record, err := iterator.mapRecord(ctx, fields, position)
		if err != nil {
			if ctx.Err() != nil {
				return sourcecontract.Item{}, ctx.Err()
			}
			if handledErr := iterator.handleMalformed(ctx, MalformedRecord{
				Position: position,
				Fields:   slices.Clone(fields),
				Err:      err,
			}); handledErr != nil {
				return sourcecontract.Item{}, handledErr
			}
			continue
		}
		iterator.lastPosition = position
		return sourcecontract.Item{
			Record:    record,
			SizeBytes: estimateRecordSize(record),
		}, nil
	}
}

func (iterator *Iterator) Close() error {
	if iterator.closed {
		return iterator.lastCloseErr
	}
	iterator.closed = true
	iterator.lastCloseErr = iterator.closeCurrent()
	return iterator.lastCloseErr
}

func (iterator *Iterator) Telemetry() sourcecontract.Telemetry {
	return iterator.telemetry.snapshot()
}

type profileCountingReader struct {
	input   io.Reader
	observe func(int64) error
}

func (reader *profileCountingReader) Read(output []byte) (int, error) {
	count, err := reader.input.Read(output)
	if count > 0 {
		if observeErr := reader.observe(int64(count)); observeErr != nil {
			return count, observeErr
		}
	}
	return count, err
}

func (iterator *Iterator) openCurrent(ctx context.Context) error {
	if err := iterator.ensureManifest(ctx); err != nil {
		return err
	}
	mapping := iterator.mappings[iterator.mappingIndex]
	file, err := os.Open(mapping.path)
	if err != nil {
		return fmt.Errorf("open CSV source %q: %w", mapping.path, err)
	}
	if iterator.options.ProfileBudget == nil {
		actualFingerprint, err := fingerprintFile(
			ctx,
			file,
			mapping.path,
			mapping.fingerprintInput,
		)
		if err != nil {
			_ = file.Close()
			return err
		}
		if mapping.fingerprint != "" && mapping.fingerprint != actualFingerprint {
			_ = file.Close()
			return errors.New("CSV source fingerprint changed while opening source")
		}
		mapping.fingerprint = actualFingerprint
		iterator.mappings[iterator.mappingIndex].fingerprint = actualFingerprint
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			_ = file.Close()
			return fmt.Errorf("rewind CSV source %q: %w", mapping.path, err)
		}
	}
	if err := iterator.options.ProfileBudget.Full(); err != nil {
		_ = file.Close()
		return err
	}
	iterator.telemetry.page()
	if err := iterator.options.ProfileBudget.Charge(
		sourcecontract.ProfileBudgetUsage{Pages: 1},
	); err != nil {
		_ = file.Close()
		return err
	}
	rawInput := io.Reader(&profileCountingReader{
		input: file,
		observe: func(bytes int64) error {
			iterator.telemetry.raw(bytes)
			return iterator.options.ProfileBudget.Charge(
				sourcecontract.ProfileBudgetUsage{RawInputBytes: bytes},
			)
		},
	})
	buffered := bufio.NewReader(rawInput)
	input := io.Reader(buffered)
	var gzipReader *gzip.Reader
	header, peekErr := buffered.Peek(2)
	if peekErr == nil && header[0] == 0x1f && header[1] == 0x8b {
		gzipReader, err = gzip.NewReader(buffered)
		if err != nil {
			_ = file.Close()
			return fmt.Errorf("open gzip CSV source %q: %w", mapping.path, err)
		}
		input = gzipReader
	} else if peekErr != nil && !errors.Is(peekErr, io.EOF) {
		_ = file.Close()
		return fmt.Errorf("inspect CSV source %q: %w", mapping.path, peekErr)
	}

	delimiter, quote, escape, err := formatRunes(mapping.format)
	if err != nil {
		if gzipReader != nil {
			_ = gzipReader.Close()
		}
		_ = file.Close()
		return err
	}
	parser, err := NewParser(input, ParserOptions{
		Delimiter:      delimiter,
		Quote:          quote,
		Escape:         escape,
		Resource:       mapping.path,
		MaxRecordBytes: iterator.options.MaxRecordBytes,
		MaxFields:      iterator.options.MaxFields,
		OptimizeRFC4180: iterator.options.OptimizeRFC4180 &&
			quote == '"' && escape == '"',
		OnInputBytes: func(bytes int64) error {
			iterator.telemetry.decoded(bytes)
			return iterator.options.ProfileBudget.Charge(
				sourcecontract.ProfileBudgetUsage{
					Rows: 0, DecodedInputBytes: bytes,
				},
			)
		},
	})
	if err != nil {
		if gzipReader != nil {
			_ = gzipReader.Close()
		}
		_ = file.Close()
		return err
	}
	current := &openMapping{
		file:    file,
		gzip:    gzipReader,
		parser:  parser,
		mapping: mapping,
	}
	if err := current.compile(
		ctx,
		iterator.options.MaxFields,
		iterator.options.MaxProperties,
	); err != nil {
		_ = current.close()
		return err
	}
	if iterator.hasResume && iterator.mappingIndex == iterator.resume.Mapping {
		if iterator.resume.Rejected > iterator.options.RejectLimit {
			_ = current.close()
			return errors.New("CSV resume token exceeds configured reject limit")
		}
		iterator.rejected = iterator.resume.Rejected
		for current.recordIndex < iterator.resume.Record {
			if err := ctx.Err(); err != nil {
				_ = current.close()
				return err
			}
			if _, _, err := parser.ReadRecord(ctx); err != nil {
				_ = current.close()
				if errors.Is(err, io.EOF) {
					return errors.New("CSV resume token is beyond end of source")
				}
				return fmt.Errorf("replay CSV checkpoint: %w", err)
			}
			current.recordIndex++
		}
		iterator.hasResume = false
	}
	iterator.current = current
	return nil
}

func (iterator *Iterator) ensureManifest(ctx context.Context) error {
	if iterator.manifestSet {
		return nil
	}
	if iterator.options.ProfileBudget != nil {
		iterator.manifest = "bounded-profile"
		iterator.manifestSet = true
		return nil
	}
	manifest, err := bindManifest(ctx, iterator.mappings)
	if err != nil {
		return err
	}
	if iterator.hasResume && iterator.resume.Fingerprint != manifest {
		return errors.New("CSV source manifest fingerprint changed since checkpoint")
	}
	iterator.manifest = manifest
	iterator.manifestSet = true
	return nil
}

func (current *openMapping) compile(
	ctx context.Context,
	maxFields int,
	maxProperties int,
) error {
	headerEnabled := current.mapping.format.Header == nil || *current.mapping.format.Header
	var columns map[string]int
	if headerEnabled {
		header, _, err := current.parser.ReadRecord(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return fmt.Errorf("CSV source %q has no header", current.mapping.path)
			}
			return fmt.Errorf("read CSV header %q: %w", current.mapping.path, err)
		}
		columns = make(map[string]int, len(header))
		for index, name := range header {
			if name == "" {
				return fmt.Errorf("CSV source %q has an empty header", current.mapping.path)
			}
			if _, exists := columns[name]; exists {
				return fmt.Errorf("CSV source %q has duplicate header %q", current.mapping.path, name)
			}
			columns[name] = index
		}
		current.compiled.fieldCount = len(header)
		current.compiled.exactFields = true
	} else {
		columns = nil
	}
	resolve := func(name string) (int, error) {
		if columns != nil {
			index, exists := columns[name]
			if !exists {
				return 0, fmt.Errorf("CSV source %q has no column %q", current.mapping.path, name)
			}
			return index, nil
		}
		index, err := strconv.Atoi(name)
		if err != nil || index < 0 || index >= maxFields {
			return 0, fmt.Errorf(
				"headerless CSV column %q must be a zero-based index below %d",
				name,
				maxFields,
			)
		}
		if index+1 > current.compiled.fieldCount {
			current.compiled.fieldCount = index + 1
		}
		return index, nil
	}

	var err error
	current.compiled.id = -1
	current.compiled.externalID = -1
	current.compiled.start = -1
	current.compiled.end = -1
	if current.mapping.kind == vertexMapping {
		current.compiled.id, err = resolve(current.mapping.idColumn)
	} else {
		if current.mapping.externalIDColumn != "" {
			current.compiled.externalID, err = resolve(current.mapping.externalIDColumn)
		}
		if err == nil {
			current.compiled.start, err = resolve(current.mapping.start.Field)
		}
		if err == nil {
			current.compiled.end, err = resolve(current.mapping.end.Field)
		}
	}
	if err != nil {
		return err
	}
	names := make([]string, 0, len(current.mapping.properties))
	if len(current.mapping.properties) > maxProperties {
		return fmt.Errorf(
			"CSV mapping has %d properties, maximum is %d",
			len(current.mapping.properties),
			maxProperties,
		)
	}
	for name := range current.mapping.properties {
		if err := ctx.Err(); err != nil {
			return err
		}
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return err
		}
		index, err := resolve(current.mapping.properties[name])
		if err != nil {
			return err
		}
		encodedName, err := jsontext.AppendQuote(nil, name)
		if err != nil {
			return fmt.Errorf("encode CSV property name %q: %w", name, err)
		}
		current.compiled.properties = append(
			current.compiled.properties,
			compiledProperty{name: name, encodedName: encodedName, index: index},
		)
	}
	return nil
}

func (iterator *Iterator) mapRecord(
	ctx context.Context,
	fields []string,
	position model.SourcePosition,
) (model.Record, error) {
	current := iterator.current
	nullValue := ""
	if current.mapping.format.NullValue != nil {
		nullValue = *current.mapping.format.NullValue
	}
	var (
		properties        model.Properties
		encodedProperties []byte
	)
	if iterator.options.PreencodeProperties {
		var err error
		encodedProperties, err = encodeCSVProperties(
			ctx,
			current.compiled.properties,
			fields,
			nullValue,
		)
		if err != nil {
			return model.Record{}, err
		}
	} else {
		properties = make(model.Properties, len(current.compiled.properties))
		for _, property := range current.compiled.properties {
			if err := ctx.Err(); err != nil {
				return model.Record{}, err
			}
			value := fields[property.index]
			if value == nullValue {
				properties[property.name] = model.Value{Kind: model.ValueNull}
			} else {
				properties[property.name] = model.Value{Kind: model.ValueString, String: value}
			}
		}
	}
	if current.mapping.kind == vertexMapping {
		externalID := fields[current.compiled.id]
		if externalID == "" || externalID == nullValue {
			return model.Record{}, errors.New("CSV vertex external ID must not be null or empty")
		}
		return model.VertexRecord(model.Vertex{
			Label:      current.mapping.label,
			Namespace:  current.mapping.namespace,
			ExternalID: model.ExternalID(externalID),
			Properties: properties, EncodedProperties: encodedProperties,
			Position: position,
		}), nil
	}

	startID := fields[current.compiled.start]
	endID := fields[current.compiled.end]
	if startID == "" || startID == nullValue || endID == "" || endID == nullValue {
		return model.Record{}, errors.New("CSV edge endpoints must not be null or empty")
	}
	externalID := ""
	if current.compiled.externalID >= 0 {
		externalID = fields[current.compiled.externalID]
		if externalID == "" || externalID == nullValue {
			return model.Record{}, errors.New("CSV edge external ID must not be null or empty")
		}
	}
	startNamespace := current.mapping.start.Namespace
	if startNamespace == "" {
		startNamespace = string(current.mapping.namespace)
	}
	endNamespace := current.mapping.end.Namespace
	if endNamespace == "" {
		endNamespace = string(current.mapping.namespace)
	}
	return model.EdgeRecord(model.Edge{
		Label:      current.mapping.label,
		Namespace:  current.mapping.namespace,
		ExternalID: model.ExternalID(externalID),
		Start: model.Endpoint{
			Label:      model.Label(current.mapping.start.Label),
			Namespace:  model.Namespace(startNamespace),
			ExternalID: model.ExternalID(startID),
		},
		End: model.Endpoint{
			Label:      model.Label(current.mapping.end.Label),
			Namespace:  model.Namespace(endNamespace),
			ExternalID: model.ExternalID(endID),
		},
		Properties: properties, EncodedProperties: encodedProperties,
		Position: position,
	}), nil
}

func encodeCSVProperties(
	ctx context.Context,
	properties []compiledProperty,
	fields []string,
	nullValue string,
) ([]byte, error) {
	output := make([]byte, 0, 2+len(properties)*16)
	output = append(output, '{')
	for index, property := range properties {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if index > 0 {
			output = append(output, ',')
		}
		output = append(output, property.encodedName...)
		output = append(output, ':')
		value := fields[property.index]
		if value == nullValue {
			output = append(output, "null"...)
			continue
		}
		var err error
		output, err = jsontext.AppendQuote(output, value)
		if err != nil {
			return nil, fmt.Errorf("encode CSV property %q: %w", property.name, err)
		}
	}
	return append(output, '}'), nil
}

func (iterator *Iterator) handleMalformed(
	ctx context.Context,
	malformed MalformedRecord,
) error {
	if iterator.options.OnMalformed == nil {
		return malformed.Err
	}
	iterator.rejected++
	if iterator.rejected > iterator.options.RejectLimit {
		return fmt.Errorf(
			"CSV reject limit %d exceeded: %w",
			iterator.options.RejectLimit,
			malformed.Err,
		)
	}
	malformed.Position.Token = formatResumeToken(resumeToken{
		Mapping:     iterator.mappingIndex,
		Record:      iterator.current.recordIndex,
		Rejected:    iterator.rejected,
		Fingerprint: iterator.manifest,
	})
	iterator.lastPosition = malformed.Position
	if err := iterator.options.OnMalformed(ctx, malformed); err != nil {
		return fmt.Errorf("write CSV quarantine record: %w", err)
	}

	return nil
}

func (iterator *Iterator) RejectionCheckpoint() (int64, model.SourcePosition) {
	return int64(iterator.rejected), iterator.lastPosition
}

func (iterator *Iterator) closeCurrent() error {
	if iterator.current == nil {
		return nil
	}
	err := iterator.current.close()
	iterator.current = nil
	return err
}

func (current *openMapping) close() error {
	var errs []error
	if current.gzip != nil {
		if err := current.gzip.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close gzip CSV source: %w", err))
		}
	}
	if err := current.file.Close(); err != nil {
		errs = append(errs, fmt.Errorf("close CSV source: %w", err))
	}
	return errors.Join(errs...)
}

func (current *openMapping) verifyFingerprint(ctx context.Context) error {
	if _, err := current.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf(
			"rewind CSV source for verification %q: %w",
			current.mapping.path,
			err,
		)
	}
	actual, err := fingerprintFile(
		ctx,
		current.file,
		current.mapping.path,
		current.mapping.fingerprintInput,
	)
	if err != nil {
		return err
	}
	if actual != current.mapping.fingerprint {
		return fmt.Errorf("CSV source %q changed during iteration", current.mapping.path)
	}
	return nil
}

func buildMappings(ctx context.Context, options IteratorOptions) ([]fileMapping, error) {
	mappings := make(
		[]fileMapping,
		0,
		len(options.Source.Vertices)+len(options.Source.Edges),
	)
	for _, vertex := range options.Source.Vertices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(vertex.Properties) > options.MaxProperties {
			return nil, fmt.Errorf(
				"CSV vertex mapping has %d properties, maximum is %d",
				len(vertex.Properties),
				options.MaxProperties,
			)
		}
		format := mergeFormat(options.Source.Defaults, vertex.Format)
		mapping, err := newFileMapping(
			vertexMapping,
			vertex.Path,
			vertex.Label,
			options.Namespace,
			format,
		)
		if err != nil {
			return nil, err
		}
		mapping.idColumn = vertex.IDColumn
		mapping.properties = maps.Clone(vertex.Properties)
		if err := bindFingerprint(&mapping); err != nil {
			return nil, err
		}
		mappings = append(mappings, mapping)
	}
	for _, edge := range options.Source.Edges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(edge.Properties) > options.MaxProperties {
			return nil, fmt.Errorf(
				"CSV edge mapping has %d properties, maximum is %d",
				len(edge.Properties),
				options.MaxProperties,
			)
		}
		format := mergeFormat(options.Source.Defaults, edge.Format)
		mapping, err := newFileMapping(
			edgeMapping,
			edge.Path,
			edge.Label,
			options.Namespace,
			format,
		)
		if err != nil {
			return nil, err
		}
		mapping.externalIDColumn = edge.ExternalIDColumn
		mapping.start = edge.Start
		mapping.end = edge.End
		mapping.properties = maps.Clone(edge.Properties)
		if err := bindFingerprint(&mapping); err != nil {
			return nil, err
		}
		mappings = append(mappings, mapping)
	}
	if len(mappings) == 0 {
		return nil, errors.New("CSV source has no mappings")
	}
	return mappings, nil
}

func newFileMapping(
	kind mappingKind,
	path string,
	label string,
	namespace string,
	format config.DelimitedOptions,
) (fileMapping, error) {
	if format.Encoding != "" && !strings.EqualFold(format.Encoding, "utf-8") {
		return fileMapping{}, fmt.Errorf(
			"CSV source %q uses unsupported encoding %q",
			path,
			format.Encoding,
		)
	}
	return fileMapping{
		kind:      kind,
		path:      path,
		label:     model.Label(label),
		namespace: model.Namespace(namespace),
		format:    format,
	}, nil
}

func bindFingerprint(mapping *fileMapping) error {
	semantic, err := json.Marshal(fingerprintMapping{
		Kind:             mapping.kind,
		Path:             mapping.path,
		Label:            mapping.label,
		Namespace:        mapping.namespace,
		IDColumn:         mapping.idColumn,
		ExternalIDColumn: mapping.externalIDColumn,
		Start:            mapping.start,
		End:              mapping.end,
		Properties:       mapping.properties,
		Format:           mapping.format,
	})
	if err != nil {
		return fmt.Errorf("encode CSV mapping fingerprint: %w", err)
	}
	mapping.fingerprintInput = semantic
	return nil
}

func mergeFormat(
	defaults config.DelimitedOptions,
	override *config.DelimitedOptions,
) config.DelimitedOptions {
	format := defaults
	if override != nil {
		if override.Delimiter != "" {
			format.Delimiter = override.Delimiter
		}
		if override.Quote != "" {
			format.Quote = override.Quote
		}
		if override.Escape != "" {
			format.Escape = override.Escape
		}
		if override.Header != nil {
			format.Header = override.Header
		}
		if override.Encoding != "" {
			format.Encoding = override.Encoding
		}
		if override.NullValue != nil {
			format.NullValue = override.NullValue
		}
	}
	if format.Delimiter == "" {
		format.Delimiter = ","
	}
	if format.Quote == "" {
		format.Quote = `"`
	}
	if format.Escape == "" {
		format.Escape = format.Quote
	}
	if format.Header == nil {
		header := true
		format.Header = &header
	}
	if format.Encoding == "" {
		format.Encoding = "utf-8"
	}
	if format.NullValue == nil {
		nullValue := ""
		format.NullValue = &nullValue
	}
	header := *format.Header
	format.Header = &header
	nullValue := *format.NullValue
	format.NullValue = &nullValue
	return format
}

func formatRunes(format config.DelimitedOptions) (rune, rune, rune, error) {
	values := []string{format.Delimiter, format.Quote, format.Escape}
	runes := make([]rune, len(values))
	for index, value := range values {
		if utf8.RuneCountInString(value) != 1 {
			return 0, 0, 0, errors.New(
				"CSV delimiter, quote, and escape must each contain one rune",
			)
		}
		runes[index], _ = utf8.DecodeRuneInString(value)
	}
	return runes[0], runes[1], runes[2], nil
}

func formatResumeToken(token resumeToken) string {
	return fmt.Sprintf(
		"csv:v2:%d:%d:%d:%s",
		token.Mapping,
		token.Record,
		token.Rejected,
		token.Fingerprint,
	)
}

func parseResumeToken(value string) (resumeToken, error) {
	parts := strings.Split(value, ":")
	if len(parts) != 6 || parts[0] != "csv" || parts[1] != "v2" {
		return resumeToken{}, errors.New("invalid CSV resume token")
	}
	mapping, err := strconv.Atoi(parts[2])
	if err != nil || mapping < 0 {
		return resumeToken{}, errors.New("invalid CSV resume mapping")
	}
	record, err := strconv.ParseInt(parts[3], 10, 64)
	if err != nil || record < 0 {
		return resumeToken{}, errors.New("invalid CSV resume record")
	}
	rejected, err := strconv.Atoi(parts[4])
	if err != nil || rejected < 0 {
		return resumeToken{}, errors.New("invalid CSV resume rejection count")
	}
	if len(parts[5]) != sha256HexLength {
		return resumeToken{}, errors.New("invalid CSV resume fingerprint")
	}
	if _, err := hex.DecodeString(parts[5]); err != nil {
		return resumeToken{}, errors.New("invalid CSV resume fingerprint")
	}
	return resumeToken{
		Mapping:     mapping,
		Record:      record,
		Rejected:    rejected,
		Fingerprint: parts[5],
	}, nil
}

func positionFromError(err error, fallback model.SourcePosition) model.SourcePosition {
	var parseErr *ParseError
	if errors.As(err, &parseErr) {
		return parseErr.Position
	}
	return fallback
}

func estimateRecordSize(record model.Record) int64 {
	const (
		vertexBase     int64 = 384
		edgeBase       int64 = 640
		propertyBase   int64 = 160
		mapBase        int64 = 64
		firstMapBucket int64 = 768
	)
	var size int64
	addProperties := func(properties model.Properties) {
		size += mapBase
		if len(properties) != 0 {
			size += firstMapBucket
		}
		for name, value := range properties {
			size += propertyBase + int64(len(name)+len(value.String))
		}
	}
	switch record.Kind() {
	case model.RecordVertex:
		size = vertexBase
		size += int64(
			len(record.Vertex.Label) +
				len(record.Vertex.Namespace) +
				len(record.Vertex.ExternalID) +
				len(record.Vertex.Position.Resource) +
				len(record.Vertex.Position.Token),
		)
		addProperties(record.Vertex.Properties)
	case model.RecordEdge:
		size = edgeBase
		size += int64(
			len(record.Edge.Label) +
				len(record.Edge.Namespace) +
				len(record.Edge.ExternalID) +
				len(record.Edge.Start.Label) +
				len(record.Edge.Start.Namespace) +
				len(record.Edge.Start.ExternalID) +
				len(record.Edge.End.Label) +
				len(record.Edge.End.Namespace) +
				len(record.Edge.End.ExternalID) +
				len(record.Edge.Position.Resource) +
				len(record.Edge.Position.Token),
		)
		addProperties(record.Edge.Properties)
	}
	return size
}

const sha256HexLength = 64

var _ sourcecontract.Iterator = (*Iterator)(nil)
