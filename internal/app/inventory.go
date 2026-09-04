package app

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourceneo4j "github.com/rioriost/agefreighter/internal/source/neo4j"
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
