package app

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pggraph"
	"github.com/rioriost/agefreighter/internal/report"
	targetruntime "github.com/rioriost/agefreighter/internal/target"
)

func verifyPostgreSQLPropertyGraph(
	ctx context.Context,
	job config.LoadJob,
	jobID string,
) (meta.Job, error) {
	operationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	ctx = operationCtx
	target, adapter, err := openPropertyGraphTarget(ctx, job)
	if err != nil {
		return meta.Job{}, err
	}
	defer target.Runtime.Close()
	stored, _, _, err := inspectPropertyGraphJob(ctx, target, adapter, job, jobID)
	return stored, err
}

func propertyGraphVerificationReport(
	ctx context.Context,
	job config.LoadJob,
	jobID string,
	options VerifyOptions,
) (report.Document, error) {
	operationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	ctx = operationCtx
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	document := report.New("verify", at)
	target, adapter, err := openPropertyGraphTarget(ctx, job)
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	document.Target = propertyGraphReportTarget(adapter.Capabilities())
	stored, mapping, inspection, inspectErr := inspectPropertyGraphJob(
		ctx, target, adapter, job, jobID)
	if stored.ID == "" {
		return report.Document{}, inspectErr
	}
	document.Job = &report.Job{ID: stored.ID, ConfigFingerprint: stored.ConfigFingerprint}
	document.Checks = append(document.Checks, jobStatusCheck(stored), schemaCheck(target.Metadata))
	if inspectErr != nil {
		if errors.Is(inspectErr, pggraph.ErrIntegrity) ||
			errors.Is(inspectErr, meta.ErrGenerationMismatch) {
			document.Checks = append(document.Checks, report.Check{
				ID: "property-graph-integrity", Status: report.CheckFail,
				Summary: "PostgreSQL property graph integrity verification failed",
				Detail:  boundedReportValue(inspectErr.Error()),
			})
			document.Outcome = report.OutcomeFail
			return validatedVerificationReport(document)
		}
		return report.Document{}, inspectErr
	}
	document.Checks = append(document.Checks,
		report.Check{ID: "property-graph-mapping", Status: report.CheckPass,
			Summary: "stored property graph mapping and physical objects match"},
		report.Check{ID: "property-graph-constraints", Status: report.CheckPass,
			Summary: "identity, uniqueness, and endpoint constraints are present"},
		report.Check{ID: "property-graph-endpoints", Status: report.CheckPass,
			Summary: "all edge endpoints resolve to the configured vertex labels"},
		report.Check{ID: "property-graph-sql-pgq", Status: report.CheckPass,
			Summary: "directed and undirected GRAPH_TABLE patterns completed"},
		report.Check{ID: "property-graph-digest", Status: report.CheckPass,
			Summary: "all logical digest ranges and the canonical root match"},
		report.Check{ID: "property-graph-rejects", Status: report.CheckPass,
			Summary: "persisted rejected-record count matches the job counters"},
	)
	document.Sections = append(document.Sections,
		propertyGraphLabelSection(inspection),
		propertyGraphDigestSection(mapping, inspection.Digests),
	)
	document.Outcome = report.OutcomePass
	return validatedVerificationReport(document)
}

func inspectPropertyGraphJob(
	ctx context.Context,
	target readOnlyTarget,
	adapter *pggraph.Adapter,
	job config.LoadJob,
	jobID string,
) (meta.Job, meta.PropertyGraphGeneration, pggraph.Inspection, error) {
	if err := target.Metadata.RequireReadCompatible(); err != nil {
		return meta.Job{}, meta.PropertyGraphGeneration{}, pggraph.Inspection{}, err
	}
	stored, err := target.Store.GetJob(ctx, jobID)
	if err != nil {
		return meta.Job{}, meta.PropertyGraphGeneration{}, pggraph.Inspection{}, err
	}
	if err := validateStoredTargetIdentity(job, stored); err != nil {
		return stored, meta.PropertyGraphGeneration{}, pggraph.Inspection{}, err
	}
	if stored.Status != meta.JobCommitted {
		return stored, meta.PropertyGraphGeneration{}, pggraph.Inspection{},
			fmt.Errorf("load job %q is %s, not committed", jobID, stored.Status)
	}
	definition, err := persistedPropertyGraphDefinition(ctx, target.Store, job, stored)
	if err != nil {
		return stored, meta.PropertyGraphGeneration{}, pggraph.Inspection{}, err
	}
	inspection, err := adapter.Inspect(ctx, jobID, definition)
	if err != nil {
		return stored, inspection.Mapping, inspection, err
	}
	mapping := inspection.Mapping
	if err := inspection.Validate(); err != nil {
		return stored, mapping, inspection, err
	}
	expectedRows := mapping.DigestRows
	if stored.LoadMode == string(config.LoadCreate) || stored.LoadMode == string(config.LoadReplace) {
		expectedRows = stored.CommittedRows
	}
	if inspection.Rows != expectedRows {
		return stored, mapping, inspection, fmt.Errorf(
			"%w: property graph contains %d rows, expected %d",
			pggraph.ErrIntegrity, inspection.Rows, expectedRows)
	}
	rejects, err := target.Store.CountRejectRecords(ctx, jobID)
	if err != nil {
		return stored, mapping, inspection, err
	}
	if rejects != stored.RejectedRows {
		return stored, mapping, inspection, fmt.Errorf(
			"%w: rejected-record count is %d, expected %d",
			pggraph.ErrIntegrity, rejects, stored.RejectedRows)
	}
	expected, err := target.Store.ListPropertyGraphDigests(ctx, jobID)
	if err != nil {
		return stored, mapping, inspection, err
	}
	if mapping.DigestRangeCount != pggraph.DigestRangeCount {
		return stored, mapping, inspection, fmt.Errorf(
			"%w: digest range count is %d, expected %d",
			pggraph.ErrIntegrity, mapping.DigestRangeCount, pggraph.DigestRangeCount)
	}
	if err := pggraph.CompareDigests(
		mapping.DigestRoot, mapping.DigestRows, expected, inspection.Digests,
	); err != nil {
		return stored, mapping, inspection, err
	}
	return stored, mapping, inspection, nil
}

func openPropertyGraphTarget(
	ctx context.Context,
	job config.LoadJob,
) (readOnlyTarget, *pggraph.Adapter, error) {
	target, err := openReadOnlyTarget(ctx, job)
	if err != nil {
		return readOnlyTarget{}, nil, err
	}
	runtime, err := targetruntime.RequirePGGraph(target.Runtime)
	if err != nil {
		target.Runtime.Close()
		return readOnlyTarget{}, nil, err
	}
	return target, runtime.PGGraphAdapter(), nil
}

func propertyGraphMigrationReport(
	ctx context.Context,
	job config.LoadJob,
	jobID string,
	options ReportOptions,
) (report.Document, error) {
	operationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	ctx = operationCtx
	target, adapter, err := openPropertyGraphTarget(ctx, job)
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	if err := target.Metadata.RequireReadCompatible(); err != nil {
		return report.Document{}, err
	}
	stored, err := target.Store.GetJob(ctx, jobID)
	if err != nil {
		return report.Document{}, err
	}
	if err := validateStoredTargetIdentity(job, stored); err != nil {
		return report.Document{}, err
	}
	snapshot := migrationReportSnapshot{
		Schema: target.Metadata, Job: stored, Counts: make(map[int64]countResult),
		BatchLimit: options.LimitBatches,
	}
	snapshot.LatestBatch, err = target.Store.LatestBatch(ctx, jobID)
	if err != nil && !errors.Is(err, meta.ErrNotFound) {
		return report.Document{}, err
	}
	snapshot.LatestBatchAvailable = err == nil
	if options.LimitBatches > 0 {
		snapshot.Batches, err = target.Store.ListBatches(ctx, jobID, options.LimitBatches+1)
		if err != nil {
			return report.Document{}, err
		}
		if len(snapshot.Batches) > options.LimitBatches {
			snapshot.BatchesTruncated = true
			snapshot.Batches = snapshot.Batches[:options.LimitBatches]
		}
	}
	snapshot.Rejects, err = target.Store.ListRejectSummaries(ctx, jobID, maxReportRejectRecords)
	if err != nil {
		return report.Document{}, err
	}
	if target.Metadata.InstalledVersion >= 15 {
		snapshot.Telemetry, err = target.Store.GetConnectorTelemetry(ctx, jobID)
		if err != nil && !errors.Is(err, meta.ErrNotFound) {
			return report.Document{}, err
		}
		snapshot.TelemetryAvailable = err == nil
	}
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	document := report.New("report", at)
	document.Job = &report.Job{ID: stored.ID, ConfigFingerprint: stored.ConfigFingerprint}
	document.Target = propertyGraphReportTarget(adapter.Capabilities())
	document.Checks = append(document.Checks, jobStatusCheck(stored), schemaCheck(target.Metadata))
	document.Sections = append(document.Sections,
		jobSection(stored), schemaSection(target.Metadata), batchSection(snapshot),
		rejectSection(snapshot.Rejects), telemetrySection(snapshot),
	)
	mapping, mappingErr := target.Store.GetPropertyGraph(ctx, jobID)
	if mappingErr == nil {
		document.Sections = append(document.Sections,
			propertyGraphMappingSection(mapping))
	} else if !errors.Is(mappingErr, meta.ErrNotFound) {
		return report.Document{}, mappingErr
	}
	if options.IncludeCounts && stored.Status == meta.JobCommitted {
		definition, definitionErr := persistedPropertyGraphDefinition(
			ctx, target.Store, job, stored,
		)
		if definitionErr != nil {
			return report.Document{}, definitionErr
		}
		inspection, inspectErr := adapter.Inspect(ctx, jobID, definition)
		if inspectErr != nil {
			if !errors.Is(inspectErr, pggraph.ErrIntegrity) {
				return report.Document{}, inspectErr
			}
			document.Checks = append(document.Checks, report.Check{
				ID: "property-graph-counts", Status: report.CheckFail,
				Summary: "property graph counts could not be trusted",
				Detail:  boundedReportValue(inspectErr.Error()),
			})
		} else {
			document.Checks = append(document.Checks, report.Check{
				ID: "property-graph-counts", Status: report.CheckPass,
				Summary: "per-label property graph counts were collected",
			})
			document.Sections = append(document.Sections, propertyGraphLabelSection(inspection))
		}
	}
	document.Outcome = report.OutcomePass
	if hasStatus(document, report.CheckFail) {
		document.Outcome = report.OutcomeFail
	} else if hasStatus(document, report.CheckWarning) {
		document.Outcome = report.OutcomeIncomplete
	}
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, err
	}
	return document, nil
}

func propertyGraphDoctor(
	ctx context.Context,
	job config.LoadJob,
	options DoctorOptions,
) (report.Document, error) {
	operationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	ctx = operationCtx
	target, adapter, err := openPropertyGraphTarget(ctx, job)
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	document := report.New("doctor", at)
	document.Target = propertyGraphReportTarget(adapter.Capabilities())
	document.Checks = append(document.Checks,
		report.Check{ID: "postgresql", Status: report.CheckPass,
			Summary: "PostgreSQL 19 or newer is reachable"},
		report.Check{ID: "sql-pgq", Status: report.CheckPass,
			Summary: "SQL/PGQ property graph DDL and GRAPH_TABLE are available"},
		schemaCheck(target.Metadata),
	)
	exists, err := adapter.GraphExists(ctx, job.Target.Schema, job.Target.Graph)
	if err != nil {
		return report.Document{}, err
	}
	graphStatus := report.CheckWarning
	graphDetail := "configured graph does not exist yet; a create load may initialize it"
	if exists {
		graphStatus = report.CheckPass
		graphDetail = "configured property graph exists"
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "property-graph-object", Status: graphStatus,
		Summary: "configured property graph object was inspected", Detail: graphDetail,
	})
	document.Sections = append(document.Sections, schemaSection(target.Metadata), report.Section{
		Title: "PostgreSQL property graph target",
		Fields: []report.Field{
			passField("backend", string(meta.TargetBackendPostgreSQLPropertyGraph)),
			passField("graph", job.Target.Graph), passField("schema", job.Target.Schema),
			passField("serverVersionNumber", strconv.Itoa(adapter.Capabilities().ServerVersionNumber)),
		},
	})
	finalizeDoctor(&document)
	encoded, err := report.Render(document, report.FormatJSON)
	if err != nil {
		return report.Document{}, err
	}
	if options.Persist {
		if err := target.Metadata.RequireCurrent(); err != nil {
			return report.Document{}, fmt.Errorf("persist doctor report: %w", err)
		}
		writeCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
		defer cancel()
		_, err := target.Store.PersistDiagnostic(writeCtx, meta.DiagnosticRecord{
			Outcome: string(document.Outcome), TargetGraph: job.Target.Graph,
			PostgreSQLVersionNumber: adapter.Capabilities().ServerVersionNumber,
			MetadataSchemaVersion:   target.Metadata.InstalledVersion, Report: encoded,
		})
		if err != nil {
			return report.Document{}, err
		}
	}
	return document, nil
}

func propertyGraphDoctorHistory(
	ctx context.Context,
	job config.LoadJob,
	limit int,
	at time.Time,
) (report.Document, error) {
	operationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
	defer cancel()
	ctx = operationCtx
	target, adapter, err := openPropertyGraphTarget(ctx, job)
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	if at.IsZero() {
		at = time.Now()
	}
	document := report.New("doctor", at)
	document.Target = propertyGraphReportTarget(adapter.Capabilities())
	if target.Metadata.State != meta.SchemaCurrent {
		document.Checks = append(document.Checks, report.Check{
			ID: "diagnostic-history", Status: report.CheckUnavailable,
			Summary: "diagnostic history is unavailable",
			Detail:  "current metadata schema is required",
		})
		finalizeDoctor(&document)
		return document, nil
	}
	values, err := target.Store.ListDiagnostics(ctx, job.Target.Graph, limit)
	if err != nil {
		return report.Document{}, err
	}
	fields := make([]report.Field, 0, len(values)+1)
	fields = append(fields, passField("records", strconv.Itoa(len(values))))
	for index, value := range values {
		fields = append(fields, passField(fmt.Sprintf("record.%03d", index+1),
			fmt.Sprintf("id=%d,outcome=%s,recordedAt=%s,postgresqlVersionNumber=%d,metadataSchemaVersion=%d",
				value.ID, value.Outcome, formatTime(value.RecordedAt),
				value.PostgreSQLVersionNumber, value.MetadataSchemaVersion)))
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "diagnostic-history", Status: report.CheckPass,
		Summary: fmt.Sprintf("read %d diagnostic history records", len(values)),
	})
	document.Sections = append(document.Sections,
		report.Section{Title: "Diagnostic history", Fields: fields})
	finalizeDoctor(&document)
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, err
	}
	return document, nil
}

func propertyGraphReportTarget(capabilities pggraph.Capabilities) *report.Target {
	return &report.Target{
		PostgreSQL: report.VersionValue{
			Value: capabilities.ServerVersion, Status: report.CheckPass,
		},
		AGE: report.VersionValue{Value: "not applicable", Status: report.CheckPass},
	}
}

func propertyGraphLabelSection(inspection pggraph.Inspection) report.Section {
	fields := make([]report.Field, 0, len(inspection.Labels)+3)
	for index, label := range inspection.Labels {
		fields = append(fields, passField(fmt.Sprintf("label.%03d.%s", index+1, label.Name),
			fmt.Sprintf("kind=%s,rows=%d,primaryKeys=%d,uniqueKeys=%d,foreignKeys=%d,missingStarts=%d,missingEnds=%d",
				string(label.Kind), label.Rows, label.PrimaryKeys, label.UniqueKeys,
				label.ForeignKeys, label.MissingStarts, label.MissingEnds)))
	}
	fields = append(fields,
		passField("directedMatches", strconv.FormatInt(inspection.DirectedMatches, 10)),
		passField("undirectedMatches", strconv.FormatInt(inspection.UndirectedMatch, 10)),
		passField("rows", strconv.FormatInt(inspection.Rows, 10)),
	)
	return report.Section{Title: "PostgreSQL property graph labels", Fields: fields}
}

func propertyGraphMappingSection(mapping meta.PropertyGraphGeneration) report.Section {
	return report.Section{Title: "PostgreSQL property graph generation", Fields: []report.Field{
		passField("definitionFingerprint", mapping.DefinitionFingerprint),
		passField("digestRangeCount", strconv.Itoa(mapping.DigestRangeCount)),
		passField("digestRoot", valueOrNone(mapping.DigestRoot)),
		passField("digestRows", strconv.FormatInt(mapping.DigestRows, 10)),
		passField("graph", mapping.Graph), passField("schema", mapping.Schema),
		passField("state", string(mapping.State)),
	}}
}

func propertyGraphDigestSection(
	mapping meta.PropertyGraphGeneration,
	digests pggraph.DigestSet,
) report.Section {
	return report.Section{Title: "PostgreSQL property graph digests", Fields: []report.Field{
		passField("actualRoot", digests.Root),
		passField("expectedRoot", mapping.DigestRoot),
		passField("logicalRows", strconv.FormatInt(digests.Rows, 10)),
		passField("nonEmptyRanges", strconv.Itoa(len(digests.Ranges))),
		passField("rangeCount", strconv.Itoa(mapping.DigestRangeCount)),
	}}
}
