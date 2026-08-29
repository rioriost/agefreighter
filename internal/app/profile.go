package app

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	sourcecosmos "github.com/rioriost/agefreighter/internal/source/cosmos"
	sourcecsv "github.com/rioriost/agefreighter/internal/source/csv"
	sourceneo4j "github.com/rioriost/agefreighter/internal/source/neo4j"
	sourcepostgres "github.com/rioriost/agefreighter/internal/source/postgres"
	"github.com/rioriost/agefreighter/internal/sqlquery"
	"github.com/rioriost/agefreighter/pkg/model"
)

type ProfileMode string

const (
	ProfileSample ProfileMode = "sample"
	ProfileExact  ProfileMode = "exact"

	DefaultProfileSampleSize = 10_000
	MaxProfileSampleSize     = 100_000

	maxProfileExactRows       = 1_000_000
	maxProfileSampleBytes     = int64(64 << 20)
	maxProfileExactBytes      = int64(1 << 30)
	maxProfileSamplePages     = int64(100_000)
	maxProfileExactPages      = int64(1_000_000)
	maxProfileSampleCharge    = 1_000.0
	maxProfileExactCharge     = 10_000.0
	maxProfileLabels          = 64
	maxProfileProperties      = 256
	maxProfileDistinct        = 1_024
	maxProfileMalformedRows   = 10_000
	profileStorageGraphLow    = int64(2)
	profileStorageGraphHigh   = int64(4)
	profileIdentityLowPerRow  = int64(128)
	profileIdentityHighPerRow = int64(384)
)

var errProfileLimit = errors.New("source profile limit reached")

type ProfileOptions struct {
	Mode        ProfileMode
	SampleSize  int
	GeneratedAt time.Time
}

type profileLimits struct {
	rows          int64
	bytes         int64
	pages         int64
	requestCharge float64
}

type profileMapping struct {
	kind       model.RecordKind
	label      string
	properties []string
	start      string
	end        string
	dynamic    bool
}

type profileAccumulator struct {
	limits       profileLimits
	rows         int64
	bytes        int64
	vertices     int64
	edges        int64
	malformed    int64
	missingID    int64
	missingEnds  int64
	missingProp  int64
	otherBad     int64
	labels       map[profileLabelKey]*profileLabelStats
	properties   map[profilePropertyKey]*profilePropertyStats
	propertyRows map[profilePropertyScope]int64
	allowed      map[profilePropertyKey]struct{}
	dynamic      map[profileLabelKey]struct{}
	budget       *sourcecontract.ProfileBudget

	propertiesTruncated bool
}

type profileLabelKey struct {
	kind  model.RecordKind
	label string
	start string
	end   string
}

type profileLabelStats struct {
	rows int64
}

type profilePropertyKey struct {
	kind     model.RecordKind
	label    string
	property string
}

type profilePropertyScope struct {
	kind  model.RecordKind
	label string
}

type profilePropertyStats struct {
	observed      int64
	present       int64
	nulls         int64
	typeCounts    [7]int64
	distinct      map[[32]byte]struct{}
	distinctLimit bool
	widthCount    int64
	widthTotal    int64
	widthMin      int64
	widthMax      int64
}

type profileRun struct {
	job                 config.LoadJob
	mode                ProfileMode
	generatedAt         time.Time
	limits              profileLimits
	mappings            []profileMapping
	accumulator         *profileAccumulator
	telemetry           *sourcecontract.Telemetry
	complete            bool
	limitReason         string
	sourceError         bool
	sourceErrorDetail   string
	mappingsTruncated   bool
	propertiesTruncated bool
	deterministic       bool
	connectorMode       string
	inputBytes          int64
	inputBytesKnown     bool
	sourceTimestamp     time.Time
	timestampKnown      bool
	budget              *sourcecontract.ProfileBudget
}

func SourceProfile(
	ctx context.Context,
	path string,
	options ProfileOptions,
) (report.Document, error) {
	if options.Mode == "" {
		options.Mode = ProfileSample
	}
	if options.Mode != ProfileSample && options.Mode != ProfileExact {
		return report.Document{}, errors.New("profile mode must be sample or exact")
	}
	if options.SampleSize == 0 {
		options.SampleSize = DefaultProfileSampleSize
	}
	if options.SampleSize < 1 || options.SampleSize > MaxProfileSampleSize {
		return report.Document{}, fmt.Errorf(
			"profile sample size must be within 1..%d",
			MaxProfileSampleSize,
		)
	}
	if err := ctx.Err(); err != nil {
		return report.Document{}, err
	}
	job, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load profile configuration: %w", err)
	}
	if err := validateImplementedSource(job); err != nil {
		return report.Document{}, err
	}
	generatedAt := options.GeneratedAt
	if generatedAt.IsZero() {
		generatedAt = time.Now()
	}
	limits := profileBounds(options)
	run := profileRun{
		job:             job,
		mode:            options.Mode,
		generatedAt:     generatedAt,
		limits:          limits,
		deterministic:   profileDeterministic(job),
		connectorMode:   profileConnectorMode(job),
		sourceTimestamp: generatedAt,
		timestampKnown:  true,
	}
	run.budget = sourcecontract.NewProfileBudget(
		sourcecontract.ProfileBudgetLimits{
			Rows: limits.rows, Pages: limits.pages,
			RawInputBytes: limits.bytes, DecodedInputBytes: limits.bytes,
			RequestCharge: limits.requestCharge,
			Labels:        maxProfileLabels, Properties: maxProfileProperties,
		},
	)

	timeoutCtx, timeoutCancel := context.WithTimeout(
		ctx,
		time.Duration(job.Runtime.OperationTimeout),
	)
	defer timeoutCancel()
	opCtx, cancel := context.WithCancelCause(timeoutCtx)
	defer cancel(nil)
	resolved, err := resolveSourceBounded(opCtx, job, run.budget)
	if err != nil {
		if ctx.Err() != nil {
			return report.Document{}, ctx.Err()
		}
		if errors.Is(err, sourcecontract.ErrProfileBudget) {
			_, run.limitReason = run.budget.Snapshot()
		} else if errors.Is(err, context.DeadlineExceeded) ||
			errors.Is(opCtx.Err(), context.DeadlineExceeded) {
			run.limitReason = "time"
		} else {
			run.sourceError = true
			run.sourceErrorDetail = "bounded source mapping discovery failed; inspect protected operational logs"
		}
		return buildSourceProfile(run)
	}
	resolved, run.mappingsTruncated, run.propertiesTruncated =
		boundedProfileJob(resolved)
	run.job = resolved
	run.deterministic = profileDeterministic(resolved)
	var mappingCap, propertyCap bool
	run.mappings, mappingCap, propertyCap = profileMappings(resolved)
	run.mappingsTruncated = run.mappingsTruncated || mappingCap
	run.propertiesTruncated = run.propertiesTruncated || propertyCap
	run.accumulator = newProfileAccumulator(limits, run.mappings)
	run.accumulator.budget = run.budget
	if resolved.Source.Type == config.SourceCSV {
		_, run.sourceTimestamp, _, run.timestampKnown = profileCSVMetadata(resolved)
	}

	iterator, err := newProfileSourceIterator(opCtx, resolved, run.accumulator, cancel)
	if err != nil {
		if ctx.Err() != nil {
			return report.Document{}, ctx.Err()
		}
		if errors.Is(err, sourcecontract.ErrProfileBudget) {
			_, run.limitReason = run.budget.Snapshot()
		} else {
			run.sourceError = true
			run.sourceErrorDetail = "source connection or mapping initialization failed; inspect protected operational logs"
		}
		return buildSourceProfile(run)
	}
	readErr := consumeProfile(opCtx, iterator, &run)
	closeErr := iterator.Close()
	run.telemetry = sourceTelemetry(iterator)
	if ctx.Err() != nil {
		return report.Document{}, ctx.Err()
	}
	if readErr != nil {
		switch {
		case errors.Is(readErr, errProfileLimit),
			errors.Is(readErr, sourcecontract.ErrProfileBudget):
			_, run.limitReason = run.budget.Snapshot()
		case errors.Is(readErr, context.DeadlineExceeded),
			errors.Is(opCtx.Err(), context.DeadlineExceeded):
			run.limitReason = "time"
		default:
			run.sourceError = true
			run.sourceErrorDetail = "source iteration failed; inspect protected operational logs"
		}
	}
	if closeErr != nil {
		run.sourceError = true
		run.sourceErrorDetail = "source close failed; inspect protected operational logs"
	}
	run.propertiesTruncated = run.propertiesTruncated ||
		run.accumulator.propertiesTruncated
	return buildSourceProfile(run)
}

func profileBounds(options ProfileOptions) profileLimits {
	if options.Mode == ProfileExact {
		return profileLimits{
			rows: maxProfileExactRows, bytes: maxProfileExactBytes,
			pages: maxProfileExactPages, requestCharge: maxProfileExactCharge,
		}
	}
	return profileLimits{
		rows: int64(options.SampleSize), bytes: maxProfileSampleBytes,
		pages: maxProfileSamplePages, requestCharge: maxProfileSampleCharge,
	}
}

func newProfileAccumulator(
	limits profileLimits,
	mappings []profileMapping,
) *profileAccumulator {
	accumulator := &profileAccumulator{
		limits: limits, labels: make(map[profileLabelKey]*profileLabelStats),
		properties:   make(map[profilePropertyKey]*profilePropertyStats),
		propertyRows: make(map[profilePropertyScope]int64),
		allowed:      make(map[profilePropertyKey]struct{}),
		dynamic:      make(map[profileLabelKey]struct{}),
	}
	for _, mapping := range mappings {
		key := profileLabelKey{
			kind: mapping.kind, label: mapping.label,
			start: mapping.start, end: mapping.end,
		}
		accumulator.labels[key] = &profileLabelStats{}
		if mapping.dynamic {
			accumulator.dynamic[key] = struct{}{}
		}
		for _, property := range mapping.properties {
			propertyKey := profilePropertyKey{
				kind: mapping.kind, label: mapping.label, property: property,
			}
			accumulator.allowed[propertyKey] = struct{}{}
			if accumulator.properties[propertyKey] == nil {
				accumulator.properties[propertyKey] = &profilePropertyStats{
					distinct: make(map[[32]byte]struct{}),
				}
			}
		}
	}
	return accumulator
}

func consumeProfile(
	ctx context.Context,
	iterator sourcecontract.Iterator,
	run *profileRun,
) error {
	for {
		if err := run.budget.CanProcess(); err != nil {
			_, run.limitReason = run.budget.Snapshot()
			return err
		}
		item, err := iterator.Next(ctx)
		if errors.Is(err, io.EOF) {
			run.complete = true
			return nil
		}
		if err != nil {
			if errors.Is(context.Cause(ctx), errProfileLimit) {
				run.limitReason = "rows"
				return errProfileLimit
			}
			return err
		}
		run.accumulator.add(item.Record, profileRecordWidth(item.Record))
	}
}

func (accumulator *profileAccumulator) add(record model.Record, size int64) {
	accumulator.rows++
	accumulator.bytes += size
	var label string
	var properties model.Properties
	var key profileLabelKey
	switch record.Kind() {
	case model.RecordVertex:
		accumulator.vertices++
		label = string(record.Vertex.Label)
		properties = record.Vertex.Properties
		key = profileLabelKey{kind: model.RecordVertex, label: label}
	case model.RecordEdge:
		accumulator.edges++
		label = string(record.Edge.Label)
		properties = record.Edge.Properties
		key = profileLabelKey{
			kind: model.RecordEdge, label: label,
			start: string(record.Edge.Start.Label), end: string(record.Edge.End.Label),
		}
	default:
		accumulator.otherBad++
		return
	}
	stats := accumulator.labels[key]
	if stats == nil {
		stats = &profileLabelStats{}
		accumulator.labels[key] = stats
	}
	stats.rows++
	propertyScope := profilePropertyScope{kind: record.Kind(), label: label}
	priorPropertyRows := accumulator.propertyRows[propertyScope]
	accumulator.propertyRows[propertyScope] = priorPropertyRows + 1
	if _, dynamic := accumulator.dynamic[key]; dynamic {
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			propertyKey := profilePropertyKey{
				kind: record.Kind(), label: label, property: name,
			}
			if _, exists := accumulator.allowed[propertyKey]; exists {
				continue
			}
			if len(accumulator.properties) >= maxProfileProperties {
				accumulator.propertiesTruncated = true
				break
			}
			if err := accumulator.budget.Charge(
				sourcecontract.ProfileBudgetUsage{Properties: 1},
			); err != nil {
				accumulator.propertiesTruncated = true
				break
			}
			accumulator.allowed[propertyKey] = struct{}{}
			accumulator.properties[propertyKey] = &profilePropertyStats{
				observed: priorPropertyRows,
				distinct: make(map[[32]byte]struct{}),
			}
		}
	}
	for propertyKey := range accumulator.allowed {
		if propertyKey.kind != record.Kind() || propertyKey.label != label {
			continue
		}
		propertyStats := accumulator.properties[propertyKey]
		propertyStats.observed++
		value, found := properties[propertyKey.property]
		if !found {
			continue
		}
		propertyStats.present++
		propertyStats.typeCounts[int(value.Kind)]++
		if value.Kind == model.ValueNull {
			propertyStats.nulls++
			continue
		}
		width := profileValueWidth(value)
		propertyStats.widthCount++
		propertyStats.widthTotal = saturatingProfileAdd(propertyStats.widthTotal, width)
		if propertyStats.widthCount == 1 || width < propertyStats.widthMin {
			propertyStats.widthMin = width
		}
		if width > propertyStats.widthMax {
			propertyStats.widthMax = width
		}
		if !propertyStats.distinctLimit {
			hash := profileValueHash(value)
			propertyStats.distinct[hash] = struct{}{}
			if len(propertyStats.distinct) >= maxProfileDistinct {
				propertyStats.distinctLimit = true
			}
		}
	}
}

func (accumulator *profileAccumulator) malformedRow(err error) {
	accumulator.malformed++
	message := strings.ToLower(err.Error())
	switch {
	case strings.Contains(message, "external id"),
		strings.Contains(message, "idfield"),
		strings.Contains(message, "id field"):
		accumulator.missingID++
	case strings.Contains(message, "endpoint"),
		strings.Contains(message, "start field"),
		strings.Contains(message, "end field"):
		accumulator.missingEnds++
	case strings.Contains(message, "property"),
		strings.Contains(message, "field"):
		accumulator.missingProp++
	default:
		accumulator.otherBad++
	}
}

func newProfileSourceIterator(
	ctx context.Context,
	job config.LoadJob,
	accumulator *profileAccumulator,
	cancel context.CancelCauseFunc,
) (sourcecontract.Iterator, error) {
	onMalformed := func(err error) error {
		accumulator.malformedRow(err)
		if accumulator.malformed >= maxProfileMalformedRows {
			cancel(errProfileLimit)
			return errProfileLimit
		}
		return nil
	}
	rejectLimit := int(min(accumulator.limits.rows, int64(maxProfileMalformedRows)))
	switch job.Source.Type {
	case config.SourceCSV:
		return sourcecsv.NewIterator(ctx, sourcecsv.IteratorOptions{
			Namespace: job.Source.Namespace, Source: *job.Source.CSV,
			RejectLimit: rejectLimit, MaxProperties: 1_024,
			OnMalformed: func(_ context.Context, malformed sourcecsv.MalformedRecord) error {
				return onMalformed(malformed.Err)
			},
			OptimizeRFC4180: true,
			ProfileBudget:   accumulator.budget,
		})
	case config.SourcePostgreSQL:
		dsn, err := resolveSecret(job.Source.PostgreSQL.Connection)
		if err != nil {
			return nil, errors.New("resolve PostgreSQL source credential failed")
		}
		return sourcepostgres.NewIterator(ctx, sourcepostgres.IteratorOptions{
			Namespace: job.Source.Namespace, Source: *job.Source.PostgreSQL,
			DSN: dsn, RejectLimit: rejectLimit, MaxProperties: 1_024,
			MaxReaders: job.Runtime.MaxSourceConcurrency,
			OnMalformed: func(_ context.Context, malformed sourcepostgres.MalformedRecord) error {
				return onMalformed(malformed.Err)
			},
			ProfileBudget: accumulator.budget,
		})
	case config.SourceNeo4j:
		var password string
		var err error
		if job.Source.Neo4j.Password != nil {
			password, err = resolveSecret(*job.Source.Neo4j.Password)
			if err != nil {
				return nil, errors.New("resolve Neo4j source credential failed")
			}
		}
		client, err := sourceneo4j.NewSDKClient(
			ctx, job.Source.Neo4j.URI, job.Source.Neo4j.Database,
			job.Source.Neo4j.Username, password, job.Source.Neo4j.FetchRows,
		)
		if err != nil {
			return nil, errors.New("open Neo4j source failed")
		}
		iterator, err := sourceneo4j.NewIterator(ctx, sourceneo4j.IteratorOptions{
			Namespace: job.Source.Namespace, Source: *job.Source.Neo4j,
			Client: client, RejectLimit: rejectLimit, MaxProperties: 1_024,
			OnMalformed: func(_ context.Context, malformed sourceneo4j.MalformedRecord) error {
				return onMalformed(malformed.Err)
			},
			ProfileBudget: accumulator.budget,
		})
		if err != nil {
			return nil, errors.Join(errors.New("initialize Neo4j profile iterator failed"), client.Close())
		}
		return iterator, nil
	case config.SourceCosmos:
		client, err := sourcecosmos.NewSDKQueryClient(
			ctx, job.Source.Cosmos.Endpoint, job.Source.Cosmos.Database,
		)
		if err != nil {
			return nil, errors.New("open Cosmos source failed")
		}
		iterator, err := sourcecosmos.NewIterator(ctx, sourcecosmos.IteratorOptions{
			Namespace: job.Source.Namespace, Source: *job.Source.Cosmos,
			Client: client, RejectLimit: rejectLimit, MaxProperties: 1_024,
			OnMalformed: func(_ context.Context, malformed sourcecosmos.MalformedRecord) error {
				return onMalformed(malformed.Err)
			},
			ProfileBudget: accumulator.budget,
		})
		if err != nil {
			return nil, errors.Join(errors.New("initialize Cosmos profile iterator failed"), client.Close())
		}
		return iterator, nil
	default:
		return nil, errors.New("unsupported source profile connector")
	}
}

func profileMappings(job config.LoadJob) ([]profileMapping, bool, bool) {
	var all []profileMapping
	add := func(
		kind model.RecordKind,
		label string,
		properties map[string]string,
		start, end string,
		dynamic bool,
	) {
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		slices.Sort(names)
		all = append(all, profileMapping{
			kind: kind, label: label, properties: names, start: start, end: end,
			dynamic: dynamic,
		})
	}
	switch job.Source.Type {
	case config.SourceCSV:
		for _, mapping := range job.Source.CSV.Vertices {
			add(model.RecordVertex, mapping.Label, mapping.Properties, "", "", false)
		}
		for _, mapping := range job.Source.CSV.Edges {
			add(model.RecordEdge, mapping.Label, mapping.Properties, mapping.Start.Label, mapping.End.Label, false)
		}
	case config.SourcePostgreSQL:
		for _, mapping := range job.Source.PostgreSQL.Vertices {
			add(model.RecordVertex, mapping.Label, mapping.Properties, "", "", false)
		}
		for _, mapping := range job.Source.PostgreSQL.Edges {
			add(model.RecordEdge, mapping.Label, mapping.Properties, mapping.Start.Label, mapping.End.Label, false)
		}
	case config.SourceNeo4j:
		for _, mapping := range job.Source.Neo4j.Vertices {
			add(model.RecordVertex, mapping.Label, mapping.Properties, "", "", false)
		}
		for _, mapping := range job.Source.Neo4j.Edges {
			add(model.RecordEdge, mapping.Label, mapping.Properties, mapping.Start.Label, mapping.End.Label, false)
		}
	case config.SourceCosmos:
		for _, mapping := range job.Source.Cosmos.Vertices {
			add(
				model.RecordVertex,
				mapping.Label,
				mapping.Properties,
				"",
				"",
				mapping.DocumentFormat == config.CosmosDocumentGremlin,
			)
		}
		for _, mapping := range job.Source.Cosmos.Edges {
			add(
				model.RecordEdge,
				mapping.Label,
				mapping.Properties,
				mapping.Start.Label,
				mapping.End.Label,
				mapping.DocumentFormat == config.CosmosDocumentGremlin,
			)
		}
	}
	slices.SortFunc(all, func(left, right profileMapping) int {
		if left.kind != right.kind {
			return int(left.kind) - int(right.kind)
		}
		if compared := strings.Compare(left.label, right.label); compared != 0 {
			return compared
		}
		if compared := strings.Compare(left.start, right.start); compared != 0 {
			return compared
		}
		return strings.Compare(left.end, right.end)
	})
	mappingsTruncated := len(all) > maxProfileLabels
	if mappingsTruncated {
		all = all[:maxProfileLabels]
	}
	propertyCount := 0
	propertiesTruncated := false
	for index := range all {
		remaining := maxProfileProperties - propertyCount
		if remaining <= 0 {
			if len(all[index].properties) > 0 {
				propertiesTruncated = true
			}
			all[index].properties = nil
			continue
		}
		if len(all[index].properties) > remaining {
			all[index].properties = all[index].properties[:remaining]
			propertiesTruncated = true
		}
		propertyCount += len(all[index].properties)
	}
	return all, mappingsTruncated, propertiesTruncated
}

func profileDeterministic(job config.LoadJob) bool {
	switch job.Source.Type {
	case config.SourceCSV:
		return true
	case config.SourcePostgreSQL:
		if job.Source.PostgreSQL.ReadMode != config.PostgreSQLReadKeyset {
			return false
		}
		for _, mapping := range job.Source.PostgreSQL.Vertices {
			if mapping.KeyField == "" ||
				!sqlquery.HasTopLevelOrderByField(mapping.Query, mapping.KeyField) {
				return false
			}
		}
		for _, mapping := range job.Source.PostgreSQL.Edges {
			if mapping.KeyField == "" ||
				!sqlquery.HasTopLevelOrderByField(mapping.Query, mapping.KeyField) {
				return false
			}
		}
	case config.SourceNeo4j:
		if job.Source.Neo4j.Discovery != nil && job.Source.Neo4j.Discovery.Enabled {
			return true
		}
		for _, mapping := range job.Source.Neo4j.Vertices {
			if mapping.KeyField == "" ||
				!sqlquery.HasFinalTopLevelOrderByField(mapping.Query, mapping.KeyField) {
				return false
			}
		}
		for _, mapping := range job.Source.Neo4j.Edges {
			if mapping.KeyField == "" ||
				!sqlquery.HasFinalTopLevelOrderByField(mapping.Query, mapping.KeyField) {
				return false
			}
		}
	case config.SourceCosmos:
		// Cosmos mappings have no declared unique ordering-key contract. An
		// ORDER BY clause alone cannot establish a repeatable bounded prefix.
		return false
	}
	return true
}

func profileCSVMetadata(job config.LoadJob) (int64, time.Time, bool, bool) {
	paths := make(map[string]struct{})
	for _, mapping := range job.Source.CSV.Vertices {
		paths[mapping.Path] = struct{}{}
	}
	for _, mapping := range job.Source.CSV.Edges {
		paths[mapping.Path] = struct{}{}
	}
	var bytes int64
	var latest time.Time
	for path := range paths {
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return 0, time.Time{}, false, false
		}
		bytes = saturatingProfileAdd(bytes, info.Size())
		if info.ModTime().After(latest) {
			latest = info.ModTime()
		}
	}
	return bytes, latest, true, !latest.IsZero()
}

func boundedProfileJob(job config.LoadJob) (config.LoadJob, bool, bool) {
	remainingMappings := maxProfileLabels
	remainingProperties := maxProfileProperties
	mappingsTruncated := false
	propertiesTruncated := false
	limitProperties := func(properties map[string]string) map[string]string {
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		slices.Sort(names)
		if len(names) > remainingProperties {
			names = names[:max(remainingProperties, 0)]
			propertiesTruncated = true
		}
		result := make(map[string]string, len(names))
		for _, name := range names {
			result[name] = properties[name]
		}
		remainingProperties -= len(names)
		return result
	}
	switch job.Source.Type {
	case config.SourceCSV:
		source := *job.Source.CSV
		source.Vertices = slices.Clone(source.Vertices)
		source.Edges = slices.Clone(source.Edges)
		for index := range source.Vertices {
			source.Vertices[index].Properties = limitProperties(source.Vertices[index].Properties)
		}
		for index := range source.Edges {
			source.Edges[index].Properties = limitProperties(source.Edges[index].Properties)
		}
		source.Vertices, source.Edges, mappingsTruncated = capProfileMappings(
			source.Vertices, source.Edges, remainingMappings,
		)
		job.Source.CSV = &source
	case config.SourcePostgreSQL:
		source := *job.Source.PostgreSQL
		source.Vertices = slices.Clone(source.Vertices)
		source.Edges = slices.Clone(source.Edges)
		for index := range source.Vertices {
			source.Vertices[index].Properties = limitProperties(source.Vertices[index].Properties)
		}
		for index := range source.Edges {
			source.Edges[index].Properties = limitProperties(source.Edges[index].Properties)
		}
		source.Vertices, source.Edges, mappingsTruncated = capProfileMappings(
			source.Vertices, source.Edges, remainingMappings,
		)
		job.Source.PostgreSQL = &source
	case config.SourceNeo4j:
		source := *job.Source.Neo4j
		source.Vertices = slices.Clone(source.Vertices)
		source.Edges = slices.Clone(source.Edges)
		for index := range source.Vertices {
			source.Vertices[index].Properties = limitProperties(source.Vertices[index].Properties)
		}
		for index := range source.Edges {
			source.Edges[index].Properties = limitProperties(source.Edges[index].Properties)
		}
		source.Vertices, source.Edges, mappingsTruncated = capProfileMappings(
			source.Vertices, source.Edges, remainingMappings,
		)
		job.Source.Neo4j = &source
	case config.SourceCosmos:
		source := *job.Source.Cosmos
		source.Vertices = slices.Clone(source.Vertices)
		source.Edges = slices.Clone(source.Edges)
		for index := range source.Vertices {
			source.Vertices[index].Properties = limitProperties(source.Vertices[index].Properties)
			source.Vertices[index].Parameters = slices.Clone(source.Vertices[index].Parameters)
		}
		for index := range source.Edges {
			source.Edges[index].Properties = limitProperties(source.Edges[index].Properties)
			source.Edges[index].Parameters = slices.Clone(source.Edges[index].Parameters)
		}
		source.Vertices, source.Edges, mappingsTruncated = capProfileMappings(
			source.Vertices, source.Edges, remainingMappings,
		)
		job.Source.Cosmos = &source
	}
	return job, mappingsTruncated, propertiesTruncated
}

func capProfileMappings[V any, E any](
	vertices []V,
	edges []E,
	maximum int,
) ([]V, []E, bool) {
	total := len(vertices) + len(edges)
	if total <= maximum {
		return vertices, edges, false
	}
	if len(vertices) >= maximum {
		return vertices[:maximum], nil, true
	}
	return vertices, edges[:maximum-len(vertices)], true
}

func buildSourceProfile(run profileRun) (report.Document, error) {
	document := report.New("profile", run.generatedAt)
	document.Checks = append(document.Checks,
		report.Check{
			ID: "read-only", Status: report.CheckPass,
			Summary: "profile used source-only read paths and did not open the target",
		},
		report.Check{
			ID: "mapping-validation", Status: report.CheckPass,
			Summary: "configured source mappings passed load-job validation",
		},
		report.Check{
			ID: "source-version", Status: report.CheckUnavailable,
			Summary: "source version is not exposed by the connector iterator",
		},
	)
	if run.sourceError {
		document.Checks = append(document.Checks, report.Check{
			ID: "source-read", Status: report.CheckUnknown,
			Summary: "bounded source profiling did not complete",
			Detail:  run.sourceErrorDetail,
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "source-profile")
	} else if run.complete && !run.mappingsTruncated && !run.propertiesTruncated {
		summary := "all configured source mappings were streamed to completion"
		document.Checks = append(document.Checks, report.Check{
			ID: "source-read", Status: report.CheckPass,
			Summary: summary,
		})
	} else {
		document.Checks = append(document.Checks, report.Check{
			ID: "source-read", Status: report.CheckUnknown,
			Summary: "source profile was truncated by a configured bound",
			Detail:  "limit=" + valueOrNone(run.limitReason),
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "source-profile")
		document.Warnings = append(document.Warnings, report.Finding{
			Code:    "PROFILE_TRUNCATED",
			Message: "reported counts and statistics are lower-bound observations from a bounded prefix",
		})
	}
	if !run.deterministic {
		document.Checks = append(document.Checks, report.Check{
			ID: "sample-order", Status: report.CheckWarning,
			Summary: "a stable unique total ordering is not established",
			Detail:  "ORDER BY alone does not prove uniqueness; repeated sample profiles may select different bounded prefixes",
		})
	}
	if run.mappingsTruncated || run.propertiesTruncated {
		document.Checks = append(document.Checks, report.Check{
			ID: "report-caps", Status: report.CheckUnknown,
			Summary: "mapping facts were truncated by profile report caps",
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "mapping-facts")
		document.Warnings = append(document.Warnings, report.Finding{
			Code:    "MAPPING_FACTS_TRUNCATED",
			Message: "additional configured labels or properties are omitted from this bounded report",
		})
	}

	document.Sections = append(document.Sections,
		profileSourceSection(run),
		profileBoundsSection(run),
		profileLabelSection(run, model.RecordVertex),
		profileLabelSection(run, model.RecordEdge),
		profilePropertySection(run),
		profileSignalsSection(run),
		profileTelemetrySection(run),
		profileCapacitySection(run),
	)
	if run.accumulator != nil && run.accumulator.malformed > 0 {
		document.Checks = append(document.Checks, report.Check{
			ID: "mapping-signals", Status: report.CheckWarning,
			Summary: fmt.Sprintf(
				"%d sampled source rows could not be mapped",
				run.accumulator.malformed,
			),
			Detail: "no rejected record values or source positions are included",
		})
		document.Warnings = append(document.Warnings, report.Finding{
			Code:    "MAPPING_RISK",
			Message: "sampled rows contain missing or incompatible mapped fields",
		})
	} else if run.accumulator != nil {
		document.Checks = append(document.Checks, report.Check{
			ID: "mapping-signals", Status: report.CheckPass,
			Summary: "no malformed identity, endpoint, or property mappings were observed",
		})
	}
	if run.mode == ProfileExact {
		document.Warnings = append(document.Warnings, report.Finding{
			Code:    "SOURCE_CONSISTENCY",
			Message: "exact counts require the connector's configured snapshot or source consistency guarantees",
		})
	}
	document.Outcome = report.OutcomePass
	if hasStatus(document, report.CheckFail) || len(document.Errors) > 0 {
		document.Outcome = report.OutcomeFail
	} else if hasStatus(document, report.CheckUnknown) ||
		hasStatus(document, report.CheckUnavailable) ||
		len(document.IncompleteChecks) > 0 {
		document.Outcome = report.OutcomeIncomplete
	}
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, fmt.Errorf("validate source profile report: %w", err)
	}
	return document, nil
}

func profileSourceSection(run profileRun) report.Section {
	fields := []report.Field{
		passField("connector", string(run.job.Source.Type)),
		passField("connectorMode", run.connectorMode),
		passField("deterministicPrefix", strconv.FormatBool(run.deterministic)),
		passField("mode", string(run.mode)),
		report.Field{
			Name: "sourceVersion", Status: report.CheckUnavailable,
			Value: "not exposed by connector iterator",
		},
	}
	if run.timestampKnown {
		fields = append(fields, passField("sourceTimestamp", formatTime(run.sourceTimestamp)))
	} else {
		fields = append(fields, unavailableField("sourceTimestamp", "source timestamp is unavailable"))
	}
	if run.mode == ProfileExact {
		fields = append(fields, report.Field{
			Name:   "consistency",
			Value:  "exact only if every configured mapping completes within hard bounds and the source remains consistent",
			Status: report.CheckWarning,
		})
	} else {
		fields = append(fields, report.Field{
			Name: "consistency", Value: "bounded prefix sample; totals are not extrapolated",
			Status: report.CheckWarning,
		})
	}
	return report.Section{Title: "Source", Fields: fields}
}

func profileBoundsSection(run profileRun) report.Section {
	limits := run.limits
	return report.Section{Title: "Bounds", Fields: []report.Field{
		passField("maxBytes", strconv.FormatInt(limits.bytes, 10)),
		passField("maxLabels", strconv.Itoa(maxProfileLabels)),
		passField("maxPages", strconv.FormatInt(limits.pages, 10)),
		passField("maxProperties", strconv.Itoa(maxProfileProperties)),
		passField("maxRequestCharge", strconv.FormatFloat(limits.requestCharge, 'f', -1, 64)),
		passField("maxRows", strconv.FormatInt(limits.rows, 10)),
		passField("operationTimeout", time.Duration(run.job.Runtime.OperationTimeout).String()),
	}}
}

func profileLabelSection(run profileRun, kind model.RecordKind) report.Section {
	title := "Vertex labels"
	if kind == model.RecordEdge {
		title = "Relationship types"
	}
	section := report.Section{Title: title, Fields: []report.Field{}}
	if run.accumulator == nil {
		section.Fields = append(section.Fields, unavailableField(
			"mappings", "resolved mappings are unavailable",
		))
		return section
	}
	index := 0
	for _, mapping := range run.mappings {
		if mapping.kind != kind {
			continue
		}
		index++
		key := profileLabelKey{
			kind: kind, label: mapping.label, start: mapping.start, end: mapping.end,
		}
		stats := run.accumulator.labels[key]
		rows := int64(0)
		if stats != nil {
			rows = stats.rows
		}
		method := "observed-bounded-prefix"
		countRange := strconv.FormatInt(rows, 10) + "..unknown"
		if run.complete && !run.mappingsTruncated && !run.propertiesTruncated {
			method = "exact-stream-complete"
			countRange = profileRange(rows, rows)
		}
		value := fmt.Sprintf(
			"label=%s,sampledRows=%d,countRange=%s,countMethod=%s,configuredProperties=%d",
			mapping.label, rows, countRange, method, len(mapping.properties),
		)
		if kind == model.RecordEdge {
			value += fmt.Sprintf(",startLabel=%s,endLabel=%s", mapping.start, mapping.end)
		}
		section.Fields = append(section.Fields, passField(
			fmt.Sprintf("%03d", index),
			value,
		))
	}
	if index == 0 {
		section.Fields = append(section.Fields, passField("mappings", "none"))
	}
	return section
}

func profilePropertySection(run profileRun) report.Section {
	section := report.Section{Title: "Property observations", Fields: []report.Field{}}
	if run.accumulator == nil {
		section.Fields = append(section.Fields, unavailableField(
			"properties", "property observations are unavailable",
		))
		return section
	}
	keys := make([]profilePropertyKey, 0, len(run.accumulator.properties))
	for key := range run.accumulator.properties {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(left, right profilePropertyKey) int {
		if left.kind != right.kind {
			return int(left.kind) - int(right.kind)
		}
		if compared := strings.Compare(left.label, right.label); compared != 0 {
			return compared
		}
		return strings.Compare(left.property, right.property)
	})
	for index, key := range keys {
		stats := run.accumulator.properties[key]
		name := fmt.Sprintf("%03d", index+1)
		if stats.observed == 0 {
			section.Fields = append(section.Fields, unavailableField(
				name,
				fmt.Sprintf(
					"kind=%s,label=%s,property=%s; no rows observed",
					profileKindName(key.kind), key.label, key.property,
				),
			))
			continue
		}
		distinct := strconv.Itoa(len(stats.distinct))
		distinctMethod := "exact-within-observed-prefix"
		if stats.distinctLimit {
			distinct = ">=" + strconv.Itoa(maxProfileDistinct)
			distinctMethod = "bounded-lower-bound"
		}
		width := "unavailable"
		if stats.widthCount > 0 {
			width = fmt.Sprintf(
				"min:%d,max:%d,average:%d",
				stats.widthMin,
				stats.widthMax,
				stats.widthTotal/stats.widthCount,
			)
		}
		section.Fields = append(section.Fields, passField(name, fmt.Sprintf(
			"kind=%s,label=%s,property=%s,observed=%d,present=%d,null=%d,types=%s,distinctNonNull=%s,distinctMethod=%s,widthBytes=%s",
			profileKindName(key.kind),
			key.label,
			key.property,
			stats.observed,
			stats.present,
			stats.nulls,
			profileTypeCounts(stats.typeCounts),
			distinct,
			distinctMethod,
			width,
		)))
	}
	if len(keys) == 0 {
		section.Fields = append(section.Fields, passField("properties", "none configured"))
	}
	return section
}

func profileSignalsSection(run profileRun) report.Section {
	if run.accumulator == nil {
		return unavailableSection("Mapping signals", "signals", "source rows were not sampled")
	}
	value := run.accumulator
	return report.Section{Title: "Mapping signals", Fields: []report.Field{
		passField("malformedRows", strconv.FormatInt(value.malformed, 10)),
		passField("missingEndpointSignals", strconv.FormatInt(value.missingEnds, 10)),
		passField("missingIdentitySignals", strconv.FormatInt(value.missingID, 10)),
		passField("missingPropertySignals", strconv.FormatInt(value.missingProp, 10)),
		passField("nullPropertySignals", strconv.FormatInt(profileNulls(value), 10)),
		passField("otherMappingSignals", strconv.FormatInt(value.otherBad, 10)),
		passField("sampledBytes", strconv.FormatInt(value.bytes, 10)),
		passField("sampledEdges", strconv.FormatInt(value.edges, 10)),
		passField("sampledRows", strconv.FormatInt(value.rows, 10)),
		passField("sampledVertices", strconv.FormatInt(value.vertices, 10)),
	}}
}

func profileTelemetrySection(run profileRun) report.Section {
	fields := []report.Field{}
	usage, _ := run.budget.Snapshot()
	inputBytes := usage.RawInputBytes
	if inputBytes == 0 {
		inputBytes = usage.DecodedInputBytes
	}
	fields = append(fields,
		passField("inputBytes", strconv.FormatInt(inputBytes, 10)),
		passField("rawInputBytes", strconv.FormatInt(usage.RawInputBytes, 10)),
		passField("decodedInputBytes", strconv.FormatInt(usage.DecodedInputBytes, 10)),
	)
	if run.telemetry == nil {
		fields = append(fields,
			passField("connector", string(run.job.Source.Type)),
			passField("failedRequestAttempts", strconv.FormatInt(usage.FailedRequestAttempts, 10)),
			passField("pages", strconv.FormatInt(usage.Pages, 10)),
			passField("requestCharge", strconv.FormatFloat(usage.RequestCharge, 'f', -1, 64)),
			passField("throttledRequests", strconv.FormatInt(usage.ThrottledRequests, 10)),
		)
		return report.Section{Title: "Connector telemetry", Fields: fields}
	}
	failed := max(run.telemetry.FailedRequestAttempts, usage.FailedRequestAttempts)
	fields = append(fields,
		passField("connector", run.telemetry.Connector),
		passField("failedRequestAttempts", strconv.FormatInt(failed, 10)),
		passField("pages", strconv.FormatInt(usage.Pages, 10)),
		passField("requestCharge", strconv.FormatFloat(usage.RequestCharge, 'f', -1, 64)),
		passField("throttledRequests", strconv.FormatInt(usage.ThrottledRequests, 10)),
	)
	return report.Section{Title: "Connector telemetry", Fields: fields}
}

func profileCapacitySection(run profileRun) report.Section {
	if run.accumulator == nil ||
		(run.sourceError && run.accumulator.rows == 0) {
		return unavailableSection("Capacity indicators", "estimates", "source rows were not sampled")
	}
	rows := run.accumulator.rows
	bytes := run.accumulator.bytes
	graphLow := saturatingProfileMultiply(bytes, profileStorageGraphLow)
	graphHigh := saturatingProfileMultiply(bytes, profileStorageGraphHigh)
	identityLow := saturatingProfileMultiply(rows, profileIdentityLowPerRow)
	identityHigh := saturatingProfileMultiply(rows, profileIdentityHighPerRow)
	stagingLow, stagingHigh := bytes, saturatingProfileMultiply(bytes, 2)
	walLow, walHigh := graphLow, saturatingProfileMultiply(graphHigh, 2)
	shadowLow, shadowHigh := int64(0), int64(0)
	backupLow, backupHigh := int64(0), int64(0)
	if run.job.Target.Mode == config.LoadReplace {
		shadowLow = saturatingProfileAdd(graphLow, identityLow)
		shadowHigh = saturatingProfileAdd(graphHigh, identityHigh)
		backupLow, backupHigh = shadowLow, shadowHigh
	}
	totalLow := profileSum(graphLow, identityLow, stagingLow, walLow, shadowLow, backupLow)
	totalHigh := profileSum(graphHigh, identityHigh, stagingHigh, walHigh, shadowHigh, backupHigh)
	method := "sampled-lower-bound-range"
	rowValue := ">=" + strconv.FormatInt(rows, 10)
	if run.complete && !run.mappingsTruncated && !run.propertiesTruncated {
		method = "complete-stream-range"
		rowValue = strconv.FormatInt(rows, 10)
	}
	return report.Section{Title: "Capacity indicators", Fields: []report.Field{
		passField("estimatedGraphBytesRange", profileRange(graphLow, graphHigh)),
		passField("estimatedIdentityBytesRange", profileRange(identityLow, identityHigh)),
		unavailableField("estimatedMigrationTime", "no user-selected or recorded throughput baseline is available"),
		passField("estimatedStagingBytesRange", profileRange(stagingLow, stagingHigh)),
		passField("estimatedTargetRows", rowValue),
		passField("estimatedWALBytesRange", profileRange(walLow, walHigh)),
		passField("method", method),
		passField("recommendedStorageBytesRange", profileRange(totalLow, totalHigh)),
		passField("replacementBackupBytesRange", profileRange(backupLow, backupHigh)),
		passField("replacementShadowBytesRange", profileRange(shadowLow, shadowHigh)),
	}}
}

func profileConnectorMode(job config.LoadJob) string {
	switch job.Source.Type {
	case config.SourceCSV:
		return "delimited"
	case config.SourcePostgreSQL:
		return string(job.Source.PostgreSQL.ReadMode)
	case config.SourceNeo4j:
		if job.Source.Neo4j.Discovery != nil && job.Source.Neo4j.Discovery.Enabled {
			return "discovered-cypher"
		}
		return "configured-cypher"
	case config.SourceCosmos:
		if job.Source.Cosmos.Gremlin != nil && job.Source.Cosmos.Gremlin.Enabled {
			return "gremlin-nosql"
		}
		return "nosql"
	default:
		return "unknown"
	}
}

func profileNulls(accumulator *profileAccumulator) int64 {
	var total int64
	for _, stats := range accumulator.properties {
		total = saturatingProfileAdd(total, stats.nulls)
	}
	return total
}

func profileKindName(kind model.RecordKind) string {
	if kind == model.RecordEdge {
		return "edge"
	}
	return "vertex"
}

func profileTypeCounts(counts [7]int64) string {
	names := [...]string{"null", "boolean", "integer", "float", "string", "list", "object"}
	parts := make([]string, 0, len(names))
	for index, count := range counts {
		if count > 0 {
			parts = append(parts, names[index]+":"+strconv.FormatInt(count, 10))
		}
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, "|")
}

func profileValueHash(value model.Value) [32]byte {
	hash := sha256.New()
	writeProfileValue(hash, value)
	var result [32]byte
	copy(result[:], hash.Sum(nil))
	return result
}

type profileHashWriter interface {
	Write([]byte) (int, error)
}

func writeProfileValue(output profileHashWriter, value model.Value) {
	_, _ = output.Write([]byte{byte(value.Kind)})
	writeLength := func(length uint64) {
		var encoded [8]byte
		binary.BigEndian.PutUint64(encoded[:], length)
		_, _ = output.Write(encoded[:])
	}
	switch value.Kind {
	case model.ValueBoolean:
		if value.Boolean {
			_, _ = output.Write([]byte{1})
		} else {
			_, _ = output.Write([]byte{0})
		}
	case model.ValueInteger:
		var encoded [8]byte
		binary.BigEndian.PutUint64(encoded[:], uint64(value.Integer))
		_, _ = output.Write(encoded[:])
	case model.ValueFloat:
		var encoded [8]byte
		binary.BigEndian.PutUint64(encoded[:], math.Float64bits(value.Float))
		_, _ = output.Write(encoded[:])
	case model.ValueString:
		writeLength(uint64(len(value.String)))
		_, _ = output.Write([]byte(value.String))
	case model.ValueList:
		writeLength(uint64(len(value.List)))
		for _, item := range value.List {
			writeProfileValue(output, item)
		}
	case model.ValueObject:
		keys := make([]string, 0, len(value.Object))
		for key := range value.Object {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		writeLength(uint64(len(keys)))
		for _, key := range keys {
			writeLength(uint64(len(key)))
			_, _ = output.Write([]byte(key))
			writeProfileValue(output, value.Object[key])
		}
	}
}

func profileValueWidth(value model.Value) int64 {
	switch value.Kind {
	case model.ValueNull:
		return 0
	case model.ValueBoolean:
		return 1
	case model.ValueInteger, model.ValueFloat:
		return 8
	case model.ValueString:
		return int64(len(value.String))
	case model.ValueList:
		total := int64(2)
		for _, item := range value.List {
			total = saturatingProfileAdd(total, profileValueWidth(item))
		}

		return total
	case model.ValueObject:
		total := int64(2)
		for key, item := range value.Object {
			total = saturatingProfileAdd(total, int64(len(key)))
			total = saturatingProfileAdd(total, profileValueWidth(item))
		}
		return total
	default:
		return 0
	}
}

func profileRecordWidth(record model.Record) int64 {
	var width int64
	var properties model.Properties
	switch record.Kind() {
	case model.RecordVertex:
		width = int64(
			len(record.Vertex.Label) +
				len(record.Vertex.Namespace) +
				len(record.Vertex.ExternalID),
		)
		properties = record.Vertex.Properties
	case model.RecordEdge:
		width = int64(
			len(record.Edge.Label) +
				len(record.Edge.Namespace) +
				len(record.Edge.ExternalID) +
				len(record.Edge.Start.Label) +
				len(record.Edge.Start.Namespace) +
				len(record.Edge.Start.ExternalID) +
				len(record.Edge.End.Label) +
				len(record.Edge.End.Namespace) +
				len(record.Edge.End.ExternalID),
		)
		properties = record.Edge.Properties
	}
	for name, value := range properties {
		width = saturatingProfileAdd(width, int64(len(name)))
		width = saturatingProfileAdd(width, profileValueWidth(value))
	}
	return width
}

func profileRange(low, high int64) string {
	return strconv.FormatInt(low, 10) + ".." + strconv.FormatInt(high, 10)
}

func profileSum(values ...int64) int64 {
	var total int64
	for _, value := range values {
		total = saturatingProfileAdd(total, value)
	}
	return total
}

func saturatingProfileAdd(left, right int64) int64 {
	if right > 0 && left > math.MaxInt64-right {
		return math.MaxInt64
	}
	return left + right
}

func saturatingProfileMultiply(left, right int64) int64 {
	if left == 0 || right == 0 {
		return 0
	}
	if left > math.MaxInt64/right {
		return math.MaxInt64
	}
	return left * right
}
