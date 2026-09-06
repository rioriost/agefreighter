package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	sourceneo4j "github.com/rioriost/agefreighter/internal/source/neo4j"
	"github.com/rioriost/agefreighter/pkg/model"
)

type InventoryOptions struct {
	GeneratedAt time.Time
}

func SourceInventory(
	ctx context.Context,
	path string,
	options InventoryOptions,
) (report.Document, error) {
	job, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load inventory configuration: %w", err)
	}
	if job.Source.Type == config.SourceCSV && job.Source.CSV != nil {
		return csvSourceInventory(ctx, job, options)
	}
	if job.Source.Type == config.SourcePostgreSQL || job.Source.Type == config.SourceCosmos {
		return networkSourceInventory(ctx, job, options)
	}
	if job.Source.Type != config.SourceNeo4j || job.Source.Neo4j == nil {
		return report.Document{}, fmt.Errorf(
			"source inventory is not implemented for %q",
			job.Source.Type,
		)
	}
	source := job.Source.Neo4j
	var password string
	if source.Password != nil {
		password, err = resolveSecret(*source.Password)
		if err != nil {
			return report.Document{}, fmt.Errorf("resolve Neo4j source password: %w", err)
		}
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	client, err := sourceneo4j.NewSDKClient(
		timeoutCtx,
		source.URI,
		source.Database,
		source.Username,
		password,
		source.FetchRows,
	)
	if err != nil {
		return report.Document{}, err
	}
	inventory, inventoryErr := sourceneo4j.CountInventory(timeoutCtx, client)
	closeErr := client.Close()
	if err := errors.Join(inventoryErr, closeErr); err != nil {
		return report.Document{}, err
	}
	generatedAt := options.GeneratedAt
	if generatedAt.IsZero() {
		generatedAt = time.Now()
	}
	document := report.New("inventory", generatedAt)
	document.Outcome = report.OutcomePass
	document.Checks = append(document.Checks, report.Check{
		ID:      "source-counts",
		Status:  report.CheckPass,
		Summary: "Neo4j returned exact transactional count-store totals",
	})
	document.Sections = append(document.Sections, report.Section{
		Title: "Source inventory",
		Fields: []report.Field{
			{Name: "connector", Value: "neo4j", Status: report.CheckPass},
			{Name: "countMethod", Value: "neo4j-transactional-count-store", Status: report.CheckPass},
			{Name: "vertices", Value: strconv.FormatInt(inventory.Vertices, 10), Status: report.CheckPass},
			{Name: "edges", Value: strconv.FormatInt(inventory.Edges, 10), Status: report.CheckPass},
			{Name: "totalRows", Value: strconv.FormatInt(inventory.TotalRows(), 10), Status: report.CheckPass},
		},
	})
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, fmt.Errorf("validate inventory report: %w", err)
	}
	return document, nil
}

// Network inventories deliberately consume every configured mapping. This is
// an explicit, separately approved read: a bounded profile cannot be promoted
// into exact target-sizing evidence. PostgreSQL holds one exported
// repeatable-read snapshot; Cosmos requires the documented source-immutability
// window because the service has no cross-container snapshot.
func networkSourceInventory(ctx context.Context, job config.LoadJob, options InventoryOptions) (report.Document, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	resolved, err := resolveSource(ctx, job)
	if err != nil {
		return report.Document{}, errors.New("network inventory mapping resolution failed")
	}
	// Inventory never quarantines malformed rows. A single malformed mapped
	// record invalidates exact evidence even if the eventual LoadJob has a
	// different error policy.
	resolved.Errors.MalformedRecord = config.MalformedFail
	resolved.Errors.RejectLimit = 0
	iterator, err := newSourceIterator(ctx, resolved, "", nil)
	if err != nil {
		return report.Document{}, errors.New("network inventory initialization failed")
	}
	return consumeNetworkInventory(ctx, resolved, iterator, options)
}

func consumeNetworkInventory(ctx context.Context, job config.LoadJob, iterator sourcecontract.Iterator, options InventoryOptions) (report.Document, error) {
	defer iterator.Close()
	acc := &profileAccumulator{}
	counts := map[string]int64{}
	for _, mapping := range profileMappingsForInventory(job) {
		counts[mapping] = 0
	}
	for {
		item, readErr := iterator.Next(ctx)
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return report.Document{}, errors.New("network inventory did not reach EOF without malformed or changed records")
		}
		if acc.rows >= csvInventoryMaxRows {
			return report.Document{}, errors.New("network inventory row limit exceeded; no exact result")
		}
		width := profileRecordWidth(item.Record)
		if width < 0 || acc.bytes > csvInventoryMaxBytes-width {
			return report.Document{}, errors.New("network inventory decoded-byte limit exceeded; no exact result")
		}
		acc.rows++
		acc.bytes += width
		switch item.Record.Kind() {
		case model.RecordVertex:
			acc.vertices++
			counts["vertex:"+string(item.Record.Vertex.Label)]++
		case model.RecordEdge:
			acc.edges++
			counts["edge:"+string(item.Record.Edge.Label)]++
		default:
			return report.Document{}, errors.New("network inventory received an invalid record")
		}
	}
	if err := iterator.Close(); err != nil {
		return report.Document{}, errors.New("network inventory source close failed")
	}
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	connector, method, summary := "postgresql", "postgresql-repeatable-read-complete-stream", "all configured PostgreSQL mappings reached EOF in one repeatable-read snapshot"
	if job.Source.Type == config.SourceCosmos {
		connector, method, summary = "cosmos-nosql", "cosmos-nosql-complete-stream", "all configured Cosmos mappings reached EOF during the required immutable-source window"
	}
	doc := report.New("inventory", at)
	doc.Outcome = report.OutcomePass
	doc.Checks = []report.Check{
		{ID: "source-counts", Status: report.CheckPass, Summary: summary},
		{ID: "read-only", Status: report.CheckPass, Summary: "source-only scan; target credentials and quarantine were not used"},
	}
	doc.Warnings = append(doc.Warnings, report.Finding{Code: "INVENTORY_NOT_MIGRATION_VERIFICATION", Message: "Counts describe mapped records, not unique identities or endpoint existence. Keep the source unchanged until migration verification completes."})
	doc.Sections = []report.Section{{Title: "Source inventory", Fields: []report.Field{
		passField("connector", connector), passField("countMethod", method),
		passField("vertices", strconv.FormatInt(acc.vertices, 10)), passField("edges", strconv.FormatInt(acc.edges, 10)),
		passField("totalRows", strconv.FormatInt(acc.rows, 10)), passField("mappedRecordBytes", strconv.FormatInt(acc.bytes, 10)),
	}}}
	labels := report.Section{Title: "Mapped record counts"}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		labels.Fields = append(labels.Fields, passField(key, strconv.FormatInt(counts[key], 10)))
	}
	capacity := profileCapacitySection(profileRun{job: job, accumulator: acc, complete: true})
	capacity.Fields = slices.DeleteFunc(capacity.Fields, func(field report.Field) bool { return field.Name == "estimatedMigrationTime" })
	doc.Sections = append(doc.Sections, labels, capacity)
	if _, err := report.Render(doc, report.FormatJSON); err != nil {
		return report.Document{}, err
	}
	return doc, nil
}

func profileMappingsForInventory(job config.LoadJob) []string {
	result := []string{}
	if job.Source.PostgreSQL != nil {
		for _, mapping := range job.Source.PostgreSQL.Vertices {
			result = append(result, "vertex:"+mapping.Label)
		}
		for _, mapping := range job.Source.PostgreSQL.Edges {
			result = append(result, "edge:"+mapping.Label)
		}
	}
	if job.Source.Cosmos != nil {
		for _, mapping := range job.Source.Cosmos.Vertices {
			result = append(result, "vertex:"+mapping.Label)
		}
		for _, mapping := range job.Source.Cosmos.Edges {
			result = append(result, "edge:"+mapping.Label)
		}
	}
	return result
}
