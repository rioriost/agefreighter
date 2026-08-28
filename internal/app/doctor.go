package app

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

const (
	MaxDoctorCatalogEntries = 100
	MaxDoctorHistory        = 100
	DefaultDoctorHistory    = 20
)

type DoctorOptions struct {
	Persist     bool
	GeneratedAt time.Time
}

func Doctor(
	ctx context.Context,
	path string,
	options DoctorOptions,
) (report.Document, error) {
	job, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load target configuration: %w", err)
	}
	timeout := time.Duration(job.Runtime.OperationTimeout)
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	probe, err := probeTarget(probeCtx, job)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	if err := ctx.Err(); err != nil {
		return report.Document{}, err
	}
	generatedAt := options.GeneratedAt
	if generatedAt.IsZero() {
		generatedAt = time.Now()
	}
	document := newDoctorDocument(probe, generatedAt)

	dsn, err := resolveSecret(job.Target.Connection)
	if err != nil {
		return report.Document{}, fmt.Errorf("resolve target connection: %w", err)
	}
	poolCtx, poolCancel := context.WithTimeout(ctx, timeout)
	pool, err := openDoctorPool(poolCtx, dsn)
	poolCancel()
	if err != nil {
		return report.Document{}, err
	}
	defer pool.Close()
	store, err := meta.New(pool)
	if err != nil {
		return report.Document{}, err
	}

	inspection := inspectDoctorSchema(ctx, store, timeout, &document)
	addInfrastructureChecks(ctx, pool, job.Target.Graph, inspection, timeout, &document)
	addDoctorHistoryStorageCheck(
		ctx,
		pool,
		inspection,
		timeout,
		&document,
	)
	if err := ctx.Err(); err != nil {
		return report.Document{}, err
	}

	ageUsable := probe.PostgreSQLStatus == age.ProbePass &&
		probe.AGEPresenceStatus == age.ProbePass &&
		probe.AGEVersionStatus == age.ProbePass &&
		probe.AGELoadabilityStatus == age.ProbePass
	if !ageUsable {
		addUnavailableAGEChecks(&document)
	} else {
		if err := addAGEDoctorChecks(
			ctx,
			job,
			inspection,
			timeout,
			&document,
		); err != nil {
			return report.Document{}, err
		}
	}
	if inspection.RequireReadCompatible() == nil {
		addMetadataDoctorChecks(
			ctx,
			store,
			job,
			timeout,
			ageUsable,
			&document,
		)
		if ageUsable {
			addCatalogResidueCheck(ctx, pool, timeout, &document)
		} else {
			addCheck(&document, "catalog-residue", report.CheckUnavailable,
				"residue and orphan checks are unavailable",
				"a loadable, compatible Apache AGE target is required")
		}
	} else {
		addUnavailableMetadataChecks(&document)
	}
	if err := ctx.Err(); err != nil {
		return report.Document{}, err
	}
	finalizeDoctor(&document)
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, fmt.Errorf("validate doctor report: %w", err)
	}
	if options.Persist {
		if !ageUsable {
			return report.Document{}, errors.New(
				"persist doctor report: Apache AGE is unavailable or incompatible; target was not modified",
			)
		}
		if err := inspection.RequireCurrent(); err != nil {
			return report.Document{}, fmt.Errorf(
				"persist doctor report: %w; run load or resume to upgrade metadata",
				err,
			)
		}
		if err := persistDoctor(ctx, job, probe, inspection, timeout, document); err != nil {
			return report.Document{}, err
		}
	}
	return document, nil
}

func DoctorHistory(
	ctx context.Context,
	path string,
	limit int,
	generatedAt time.Time,
) (report.Document, error) {
	if limit <= 0 || limit > MaxDoctorHistory {
		return report.Document{}, fmt.Errorf(
			"doctor history limit must be within 1..%d",
			MaxDoctorHistory,
		)
	}
	job, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load target configuration: %w", err)
	}
	timeout := time.Duration(job.Runtime.OperationTimeout)
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	probe, err := probeTarget(probeCtx, job)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	if generatedAt.IsZero() {
		generatedAt = time.Now()
	}
	document := newDoctorDocument(probe, generatedAt)
	document.Checks = nil
	document.Sections = nil

	dsn, err := resolveSecret(job.Target.Connection)
	if err != nil {
		return report.Document{}, fmt.Errorf("resolve target connection: %w", err)
	}
	openCtx, openCancel := context.WithTimeout(ctx, timeout)
	pool, err := openDoctorPool(openCtx, dsn)
	openCancel()
	if err != nil {
		return report.Document{}, err
	}
	defer pool.Close()
	store, err := meta.New(pool)
	if err != nil {
		return report.Document{}, err
	}
	readCtx, readCancel := context.WithTimeout(ctx, timeout)
	inspection, inspectErr := store.InspectSchema(readCtx)
	readCancel()
	if inspectErr != nil {
		if err := canceled(ctx, inspectErr); err != nil {
			return report.Document{}, err
		}
		addCheck(&document, "diagnostic-history", report.CheckUnknown,
			"diagnostic history could not be inspected",
			safeDatabaseDetail(inspectErr, "metadata catalog inspection failed"))
		finalizeDoctor(&document)
		return document, nil
	}
	if inspection.State == meta.SchemaNewer {
		return report.Document{}, fmt.Errorf(
			"doctor history: metadata schema version %d is newer than supported version %d",
			inspection.InstalledVersion,
			inspection.SupportedVersion,
		)
	}
	if inspection.State != meta.SchemaCurrent {
		status := report.CheckUnavailable
		if inspection.State == meta.SchemaUnknown {
			status = report.CheckUnknown
		}
		addCheck(&document, "diagnostic-history", status,
			"diagnostic history is unavailable",
			fmt.Sprintf(
				"metadata schema is %s at v%d; history requires v%d and doctor will not migrate",
				inspection.State,
				inspection.InstalledVersion,
				meta.SupportedSchemaVersion,
			))
		document.Sections = append(document.Sections, unavailableSection(
			"Diagnostic history",
			"records",
			"no history was read and the metadata schema was not changed",
		))
		finalizeDoctor(&document)
		return document, nil
	}
	readCtx, readCancel = context.WithTimeout(ctx, timeout)
	records, err := store.ListDiagnostics(readCtx, job.Target.Graph, limit)
	readCancel()
	if err != nil {
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return report.Document{}, canceledErr
		}
		addCheck(&document, "diagnostic-history", diagnosticErrorStatus(err),
			"diagnostic history could not be read",
			safeDatabaseDetail(err, "bounded diagnostic history read failed"))
		finalizeDoctor(&document)
		return document, nil
	}
	addCheck(&document, "diagnostic-history", report.CheckPass,
		fmt.Sprintf("%d persisted diagnostic reports returned", len(records)),
		fmt.Sprintf("bounded by --limit=%d and ordered newest first", limit))
	section := report.Section{Title: "Diagnostic history", Fields: []report.Field{}}
	if len(records) == 0 {
		section.Fields = append(section.Fields, passField("records", "none"))
	}
	if len(records) > 0 {
		document.GeneratedAt = records[0].RecordedAt
	}
	for index, record := range records {
		prefix := fmt.Sprintf("%03d", index+1)
		section.Fields = append(section.Fields, passField(
			prefix,
			fmt.Sprintf(
				"id=%d recordedAt=%s outcome=%s targetGraph=%s postgresqlVersionNumber=%d ageVersion=%s metadataSchemaVersion=%d",
				record.ID,
				formatTime(record.RecordedAt),
				record.Outcome,
				record.TargetGraph,
				record.PostgreSQLVersionNumber,
				record.AGEVersion,
				record.MetadataSchemaVersion,
			),
		))
	}

	document.Sections = append(document.Sections, section)
	finalizeDoctor(&document)
	return document, nil
}

func addDoctorHistoryStorageCheck(
	ctx context.Context,
	pool *pgxpool.Pool,
	inspection meta.SchemaInspection,
	timeout time.Duration,
	document *report.Document,
) {
	if inspection.State != meta.SchemaCurrent {
		status := report.CheckUnavailable
		if inspection.State == meta.SchemaUnknown {
			status = report.CheckUnknown
		}
		addCheck(document, "diagnostic-history-storage", status,
			"diagnostic history storage is unavailable",
			fmt.Sprintf(
				"metadata schema is %s at v%d; current v%d is required",
				inspection.State,
				inspection.InstalledVersion,
				inspection.SupportedVersion,
			))
		return
	}
	var hasRecords bool
	err := queryDoctorRow(ctx, pool, timeout, `
		SELECT EXISTS (
			SELECT 1
			FROM agefreighter_meta.diagnostic_history
			LIMIT 1
		)`,
	).Scan(&hasRecords)
	if err != nil {
		addClassifiedCheck(ctx, document, "diagnostic-history-storage",
			"diagnostic history storage could not be inspected", err)
		return
	}
	addCheck(document, "diagnostic-history-storage", report.CheckPass,
		"bounded diagnostic history storage is readable",
		fmt.Sprintf("has_records=%t retention_limit=%d",
			hasRecords, meta.MaxDiagnosticHistory))
}

func openDoctorPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	poolConfig, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, errors.New("parse PostgreSQL target connection")
	}
	poolConfig.MinConns = 0
	poolConfig.MaxConns = 1
	poolConfig.AfterConnect = func(ctx context.Context, connection *pgx.Conn) error {
		_, err := connection.Exec(ctx, `SET default_transaction_read_only = on`)
		return err
	}
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, errors.New("create read-only PostgreSQL diagnostic pool")
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect read-only PostgreSQL diagnostic pool: %w", err)
	}
	return pool, nil
}

func newDoctorDocument(probe age.DegradedProbe, at time.Time) report.Document {
	document := report.New("doctor", at)
	document.Target = &report.Target{
		PostgreSQL: versionValue(probe.PostgreSQLVersion, probe.PostgreSQLStatus),
		AGE:        versionValue(probe.AGEVersion, probe.AGEVersionStatus),
	}
	addProbeCheck := func(
		id, summary, detail string,
		status age.ProbeStatus,
	) {
		addCheck(&document, id, probeCheckStatus(status), summary, detail)
	}
	addProbeCheck(
		"postgresql-version",
		"PostgreSQL server compatibility was probed",
		probe.PostgreSQLDetail,
		probe.PostgreSQLStatus,
	)
	addProbeCheck(
		"age-installation",
		"Apache AGE extension installation was probed",
		probe.AGEPresenceDetail,
		probe.AGEPresenceStatus,
	)
	addProbeCheck(
		"age-version",
		"Apache AGE extension compatibility was probed",
		probe.AGEVersionDetail,
		probe.AGEVersionStatus,
	)
	addProbeCheck(
		"age-loadability",
		"Apache AGE loadability was probed",
		probe.AGELoadabilityDetail,
		probe.AGELoadabilityStatus,
	)
	switch probe.AGEPreloadStatus {
	case age.PreloadConfigured:
		addCheck(&document, "age-preload", report.CheckPass,
			"Apache AGE is configured in shared_preload_libraries", "")
	case age.PreloadNotConfigured:
		addCheck(&document, "age-preload", report.CheckWarning,
			"Apache AGE is not configured in shared_preload_libraries",
			"agefreighter will attempt bounded dynamic initialization")
	default:
		addCheck(&document, "age-preload", report.CheckUnknown,
			"Apache AGE preload configuration is unknown",
			"verify permission to read pg_settings.shared_preload_libraries")
	}
	return document
}

func inspectDoctorSchema(
	ctx context.Context,
	store *meta.Store,
	timeout time.Duration,
	document *report.Document,
) meta.SchemaInspection {
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	inspection, err := store.InspectSchema(readCtx)
	cancel()
	if err != nil {
		if ctx.Err() != nil {
			return meta.SchemaInspection{
				State:            meta.SchemaUnknown,
				SupportedVersion: meta.SupportedSchemaVersion,
				Detail:           ctx.Err().Error(),
			}
		}
		inspection = meta.SchemaInspection{
			State:            meta.SchemaUnknown,
			SupportedVersion: meta.SupportedSchemaVersion,
			Detail:           safeDatabaseDetail(err, "metadata schema inspection failed"),
		}
	}
	status := report.CheckPass
	switch inspection.State {
	case meta.SchemaCurrent:
	case meta.SchemaPending:
		status = report.CheckWarning
	case meta.SchemaAbsent:
		status = report.CheckUnavailable
	case meta.SchemaNewer, meta.SchemaInvalid:
		status = report.CheckFail
	default:
		status = report.CheckUnknown
	}
	detail := inspection.Detail
	if detail == "" {
		detail = fmt.Sprintf(
			"installed=%d supported=%d pending=%d; doctor does not migrate",
			inspection.InstalledVersion,
			inspection.SupportedVersion,
			inspection.PendingVersions,
		)
	}
	addCheck(document, "metadata-schema", status,
		fmt.Sprintf("agefreighter metadata schema is %s", inspection.State),
		detail)
	document.Sections = append(document.Sections, schemaSection(inspection))
	return inspection
}

func addAGEDoctorChecks(
	ctx context.Context,
	job config.LoadJob,
	inspection meta.SchemaInspection,
	timeout time.Duration,
	document *report.Document,
) error {
	openCtx, cancel := context.WithTimeout(ctx, timeout)
	adapter, store, err := openAGEStore(openCtx, job)
	cancel()
	if err != nil {
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return canceledErr
		}
		addCheck(document, "age-session", report.CheckFail,
			"Apache AGE session initialization failed",
			"the extension probed as loadable but a configured AGE session could not open")
		for _, value := range []struct{ id, summary string }{
			{"age-search-path", "Apache AGE search path could not be verified"},
			{"target-graph", "target graph catalog could not be verified"},
			{"graph-catalog-objects", "graph catalog objects could not be verified"},
		} {
			addCheck(document, value.id, report.CheckUnavailable,
				value.summary, "configured Apache AGE session initialization failed")
		}
		return nil
	}
	defer adapter.Close()
	readCtx, readCancel := context.WithTimeout(ctx, timeout)
	searchPath, err := adapter.CurrentSearchPath(readCtx)
	readCancel()
	if err != nil {
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return canceledErr
		}
		addCheck(document, "age-search-path", diagnosticErrorStatus(err),
			"Apache AGE search path could not be verified",
			safeDatabaseDetail(err, "search path inspection failed"))
	} else if !searchPathContains(searchPath, "ag_catalog") {
		addCheck(document, "age-search-path", report.CheckFail,
			"Apache AGE session search path omits ag_catalog",
			"configured session must include ag_catalog")
	} else {
		addCheck(document, "age-search-path", report.CheckPass,
			"Apache AGE session search path includes ag_catalog", "")
	}
	readCtx, readCancel = context.WithTimeout(ctx, timeout)
	graph, graphErr := adapter.LookupGraph(readCtx, job.Target.Graph)
	readCancel()
	if errors.Is(graphErr, age.ErrCatalogEntryNotFound) {
		status := report.CheckPass
		detail := "target graph is absent and create mode may create it"
		if job.Target.Mode != config.LoadCreate {
			status = report.CheckWarning
			detail = "target graph is absent; this mode normally expects an existing graph"
		}
		addCheck(document, "target-graph", status,
			fmt.Sprintf("target graph %q is absent", job.Target.Graph), detail)
		addCheck(document, "graph-catalog-objects", status,
			"target graph has no label, relation, or sequence objects to inspect",
			detail)
		return nil
	}
	if graphErr != nil {
		if canceledErr := canceled(ctx, graphErr); canceledErr != nil {
			return canceledErr
		}
		addCheck(document, "target-graph", catalogErrorStatus(graphErr),
			fmt.Sprintf("target graph %q could not be inspected", job.Target.Graph),
			safeDatabaseDetail(graphErr, "AGE graph catalog inspection failed"))
		addCheck(document, "graph-catalog-objects", report.CheckUnavailable,
			"target graph objects could not be inspected",
			"target graph catalog inspection did not pass")
		return nil
	}
	addCheck(document, "target-graph", report.CheckPass,
		fmt.Sprintf("target graph %q has a consistent graph and namespace catalog", graph.Name),
		fmt.Sprintf("graph_oid=%d namespace_oid=%d", graph.GraphOID, graph.NamespaceOID))
	if inspection.RequireReadCompatible() != nil {
		addCheck(document, "graph-catalog-objects", report.CheckUnavailable,
			"metadata-backed graph object checks are unavailable",
			"a read-compatible metadata schema is required")
		return nil
	}
	readCtx, readCancel = context.WithTimeout(ctx, timeout)
	generationPage, err := store.ListCurrentGraphGenerations(
		readCtx,
		job.Target.Graph,
		MaxDoctorCatalogEntries,
	)
	readCancel()
	if err != nil {
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return canceledErr
		}
		addCheck(document, "graph-catalog-objects", diagnosticErrorStatus(err),
			"metadata-backed graph object checks could not be read",
			safeDatabaseDetail(err, "graph generation catalog read failed"))
		return nil
	}
	checked := 0
	matchingGenerations := len(generationPage.Generations)
	complete := generationPage.Complete
	for _, generation := range generationPage.Generations {
		if generation.GraphOID != graph.GraphOID ||
			generation.NamespaceOID != graph.NamespaceOID {
			addCheck(document, "graph-catalog-objects", report.CheckFail,
				"target graph catalog does not match its active metadata generation",
				fmt.Sprintf(
					"metadata_graph_oid=%d catalog_graph_oid=%d metadata_namespace_oid=%d catalog_namespace_oid=%d",
					generation.GraphOID,
					graph.GraphOID,
					generation.NamespaceOID,
					graph.NamespaceOID,
				))
			return nil
		}
		readCtx, readCancel = context.WithTimeout(ctx, timeout)
		labelPage, labelErr := store.ListLabelGenerationPage(
			readCtx,
			generation.ID,
			MaxDoctorCatalogEntries,
		)
		readCancel()
		if labelErr != nil {
			addCheck(document, "graph-catalog-objects",
				diagnosticErrorStatus(labelErr),
				"label-generation metadata could not be read",
				safeDatabaseDetail(labelErr, "label generation catalog read failed"))
			return nil
		}
		if !labelPage.Complete {
			complete = false
			document.Warnings = append(document.Warnings, report.Finding{
				Code: "LABEL_CATALOG_TRUNCATED",
				Message: fmt.Sprintf(
					"graph object checks are limited to %d labels",
					MaxDoctorCatalogEntries,
				),
			})
		}
		for _, label := range labelPage.Generations {
			readCtx, readCancel = context.WithTimeout(ctx, timeout)
			catalog, lookupErr := adapter.LookupLabel(
				readCtx,
				job.Target.Graph,
				label.LabelName,
			)
			readCancel()
			if lookupErr != nil {
				if canceledErr := canceled(ctx, lookupErr); canceledErr != nil {
					return canceledErr
				}
				addCheck(document, "graph-catalog-objects",
					catalogErrorStatus(lookupErr),
					fmt.Sprintf("catalog object for label %q is inconsistent",
						label.LabelName),
					safeDatabaseDetail(lookupErr, "label catalog lookup failed"))
				return nil
			}
			if catalog.LabelID != label.LabelID ||
				catalog.RelationOID != label.RelationOID ||
				catalog.SequenceOID != label.SequenceOID ||
				byte(catalog.Kind) != byte(label.Kind) {
				addCheck(document, "graph-catalog-objects", report.CheckFail,
					fmt.Sprintf("catalog object for label %q does not match metadata",
						label.LabelName),
					fmt.Sprintf(
						"metadata_label_id=%d catalog_label_id=%d relation_oid=%d catalog_relation_oid=%d sequence_oid=%d catalog_sequence_oid=%d",
						label.LabelID,
						catalog.LabelID,
						label.RelationOID,
						catalog.RelationOID,
						label.SequenceOID,
						catalog.SequenceOID,
					))
				return nil
			}
			checked++
		}
	}
	status := report.CheckPass
	summary := "bounded graph label, relation, and sequence catalogs are consistent"
	if matchingGenerations == 0 {
		status = report.CheckWarning
		summary = "target graph exists without an active or loading metadata generation"
	} else if !complete {
		status = report.CheckUnknown
		summary = "target graph catalog consistency could not be proven within the bounded limit"
	}
	addCheck(document, "graph-catalog-objects", status,
		summary,
		fmt.Sprintf("matching_generations=%d checked_labels=%d limit=%d",
			matchingGenerations, checked, MaxDoctorCatalogEntries))
	return nil
}

func addMetadataDoctorChecks(
	ctx context.Context,
	store *meta.Store,
	job config.LoadJob,
	timeout time.Duration,
	ageUsable bool,
	document *report.Document,
) {
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	jobPage, err := store.ListActiveJobHealth(
		readCtx,
		MaxDoctorCatalogEntries,
	)
	cancel()
	if err != nil {
		addClassifiedCheck(ctx, document, "metadata-jobs",
			"bounded load-job metadata could not be read", err)
		addCheck(document, "graph-generations", report.CheckUnavailable,
			"graph-generation health is unavailable",
			"load-job metadata read did not pass")
		addCheck(document, "retained-backups", report.CheckUnavailable,
			"retained-backup health is unavailable",
			"load-job metadata read did not pass")
		return
	}
	if !jobPage.Complete {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "JOBS_TRUNCATED",
			Message: fmt.Sprintf(
				"active load-job diagnostics are limited to %d newest entries",
				MaxDoctorCatalogEntries,
			),
		})
	}
	stale := 0
	conflictingTargets := make(map[string]struct{})
	now := document.GeneratedAt
	for _, value := range jobPage.Jobs {
		if now.Sub(value.UpdatedAt) > 2*timeout {
			stale++
		}
		if value.Conflicting {
			conflictingTargets[value.TargetGraph] = struct{}{}
		}
	}
	status := report.CheckPass
	detail := fmt.Sprintf(
		"bounded_jobs=%d active_or_pending=%d stale=%d conflicting_targets=%d",
		len(jobPage.Jobs),
		len(jobPage.Jobs),
		stale,
		len(conflictingTargets),
	)
	if !jobPage.Complete {
		status = report.CheckUnknown
	} else if stale > 0 || len(conflictingTargets) > 0 {
		status = report.CheckWarning
	}
	addCheck(document, "metadata-jobs", status,
		func() string {
			if !jobPage.Complete {
				return "active load-job health could not be proven within the bounded limit"
			}
			return "active load-job state was inspected"
		}(),
		detail)

	readCtx, cancel = context.WithTimeout(ctx, timeout)
	generationPage, genErr := store.ListCurrentGraphGenerations(
		readCtx,
		job.Target.Graph,
		MaxDoctorCatalogEntries,
	)
	cancel()
	if genErr != nil {
		addClassifiedCheck(ctx, document, "graph-generations",
			"bounded graph-generation metadata could not be read", genErr)
		addCheck(document, "retained-backups", report.CheckUnavailable,
			"retained-backup health is unavailable",
			"graph-generation metadata read did not pass")
		return
	}
	if !generationPage.Complete {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "GENERATIONS_TRUNCATED",
			Message: fmt.Sprintf(
				"current target graph-generation diagnostics are limited to %d entries",
				MaxDoctorCatalogEntries,
			),
		})
	}
	counts := map[meta.GenerationState]int{}
	for _, generation := range generationPage.Generations {
		counts[generation.State]++
	}
	generationStatus := report.CheckPass
	generationSummary := "current target graph-generation states were inspected"
	if !generationPage.Complete {
		generationStatus = report.CheckUnknown
		generationSummary = "target graph-generation health could not be proven within the bounded limit"
	}
	addCheck(document, "graph-generations", generationStatus,
		generationSummary,
		fmt.Sprintf(
			"active=%d loading=%d retired=%d target_generations=%d",
			counts[meta.GenerationActive],
			counts[meta.GenerationLoading],
			counts[meta.GenerationRetired],
			len(generationPage.Generations),
		))

	readCtx, cancel = context.WithTimeout(ctx, timeout)
	backups, backupErr := store.ListRetainedBackups(
		readCtx,
		MaxDoctorCatalogEntries+1,
	)
	cancel()
	if backupErr != nil {
		addClassifiedCheck(ctx, document, "retained-backups",
			"bounded retained-backup metadata could not be read", backupErr)
		return
	}
	backupStatus := report.CheckPass
	backupCount := len(backups)
	backupsComplete := backupCount <= MaxDoctorCatalogEntries
	if !backupsComplete {
		backupCount = MaxDoctorCatalogEntries
		backupStatus = report.CheckUnknown
	} else if backupCount > 0 {
		backupStatus = report.CheckWarning
	}
	addCheck(document, "retained-backups", backupStatus,
		fmt.Sprintf("%d retained replacement backups were found in the bounded window", backupCount),
		func() string {
			if !backupsComplete {
				return "additional retained backups exist beyond the diagnostic limit"
			}
			return "review retention before using cleanup; doctor never removes objects"
		}())

}

func addInfrastructureChecks(
	ctx context.Context,
	pool *pgxpool.Pool,
	graphSchema string,
	schema meta.SchemaInspection,
	timeout time.Duration,
	document *report.Document,
) {
	var databaseSize int64
	err := queryDoctorRow(ctx, pool, timeout,
		`SELECT pg_catalog.pg_database_size(current_database())`,
	).Scan(&databaseSize)
	addValueCheck(ctx, document, "database-size", err,
		"database size visibility",
		fmt.Sprintf("bytes=%d", databaseSize))

	var walLSN string
	err = queryDoctorRow(ctx, pool, timeout,
		`SELECT pg_catalog.pg_current_wal_lsn()::text`,
	).Scan(&walLSN)
	addValueCheck(ctx, document, "wal-visibility", err,
		"WAL visibility",
		func() string {
			if walLSN == "" {
				return ""
			}
			return "current WAL position is visible"
		}())

	var autovacuum string
	err = queryDoctorRow(ctx, pool, timeout,
		`SELECT current_setting('autovacuum')`,
	).Scan(&autovacuum)
	status := report.CheckPass
	detail := "autovacuum=" + autovacuum
	if err == nil && autovacuum != "on" {
		status = report.CheckWarning
	}
	if err != nil {
		addClassifiedCheck(ctx, document, "autovacuum",
			"autovacuum setting could not be read", err)
	} else {
		addCheck(document, "autovacuum", status,
			"autovacuum configuration was inspected", detail)
	}

	var canConnect, schemaUsage bool
	err = queryDoctorRow(ctx, pool, timeout, `
		SELECT
			has_database_privilege(current_user, current_database(), 'CONNECT'),
			CASE
				WHEN pg_catalog.to_regnamespace('agefreighter_meta') IS NULL
				THEN false
				ELSE has_schema_privilege(
					current_user, 'agefreighter_meta', 'USAGE'
				)
			END`,
	).Scan(&canConnect, &schemaUsage)
	if err != nil {
		addClassifiedCheck(ctx, document, "permissions",
			"target permissions could not be inspected", err)
	} else {
		status = report.CheckPass
		if !canConnect {
			status = report.CheckFail
		} else if !schemaUsage {
			status = report.CheckWarning
		}
		addCheck(document, "permissions", status,
			"target database and metadata-schema privileges were inspected",
			fmt.Sprintf("database_connect=%t metadata_schema_usage=%t",
				canConnect, schemaUsage))
	}

	var visibleTables, concerningTables, neverAnalyzedTables int
	err = queryDoctorRow(ctx, pool, timeout, `
		SELECT COUNT(*)::integer,
		       COUNT(*) FILTER (
			       WHERE n_dead_tup > 10000
			         AND n_dead_tup > GREATEST(n_live_tup / 5, 1000)
		       )::integer,
		       COUNT(*) FILTER (
			       WHERE last_analyze IS NULL
			         AND last_autoanalyze IS NULL
			         AND n_live_tup > 0
		       )::integer
		FROM (
			SELECT n_live_tup, n_dead_tup, last_analyze, last_autoanalyze
			FROM pg_catalog.pg_stat_user_tables
			WHERE schemaname = 'agefreighter_meta'
			   OR schemaname = $1
			ORDER BY schemaname, relname
			LIMIT $2
		) bounded_stats`,
		graphSchema,
		MaxDoctorCatalogEntries+1,
	).Scan(&visibleTables, &concerningTables, &neverAnalyzedTables)
	if err != nil {
		addClassifiedCheck(ctx, document, "table-maintenance",
			"bounded table-maintenance statistics could not be read", err)
	} else {
		status = report.CheckPass
		if visibleTables > MaxDoctorCatalogEntries {
			status = report.CheckUnknown
		} else if concerningTables > 0 || neverAnalyzedTables > 0 {
			status = report.CheckWarning
		}
		addCheck(document, "table-maintenance", status,
			"bounded dead-tuple and analyze visibility was inspected",
			fmt.Sprintf(
				"visible_tables=%d concerning_dead_tuple_tables=%d never_analyzed_tables=%d",
				visibleTables,
				concerningTables,
				neverAnalyzedTables,
			))
	}

	addMetadataIndexCheck(ctx, pool, schema, timeout, document)

	addCheck(document, "free-storage", report.CheckWarning,
		"filesystem free-storage visibility is unavailable through portable PostgreSQL catalogs",
		"use platform storage monitoring; doctor does not execute host commands")
}

type metadataIndexDefinition struct {
	Name       string
	Relation   string
	Unique     bool
	Keys       []string
	KeyOptions []int16
	Include    []string
	Predicate  string
}

var requiredMetadataIndexes = []metadataIndexDefinition{
	{
		Name: "graph_generation_current_name_uq", Relation: "graph_generation",
		Unique:     true,
		Keys:       []string{"graph_name"},
		KeyOptions: []int16{0},
		Predicate:  "state = ANY (ARRAY['loading'::text, 'active'::text])",
	},
	{
		Name: "load_batch_latest_idx", Relation: "load_batch",
		Keys:       []string{"job_id", "batch_id", "attempt"},
		KeyOptions: []int16{0, 3, 3},
	},
	{
		Name: "load_batch_committed_uq", Relation: "load_batch",
		Unique:     true,
		Keys:       []string{"job_id", "batch_id"},
		KeyOptions: []int16{0, 0},
		Predicate:  "status = 'committed'::text",
	},
	{
		Name: "load_batch_running_uq", Relation: "load_batch",
		Unique:     true,
		Keys:       []string{"job_id", "batch_id"},
		KeyOptions: []int16{0, 0},
		Predicate:  "status = 'running'::text",
	},
	{
		Name: "label_generation_catalog_idx", Relation: "label_generation",
		Keys: []string{
			"graph_namespace_oid", "label_id", "relation_oid", "mapping_generation",
		},
		KeyOptions: []int16{0, 0, 0, 0},
	},
	{
		Name: "vertex_identity_lookup_uq", Relation: "vertex_identity",
		Unique: true,
		Keys: []string{
			"graph_generation_id", "source_namespace", "label_id", "external_id",
		},
		KeyOptions: []int16{0, 0, 0, 0},
		Include:    []string{"graph_id", "label_generation_id"},
	},
	{
		Name: "vertex_identity_graph_id_uq", Relation: "vertex_identity",
		Unique:     true,
		Keys:       []string{"graph_generation_id", "graph_id"},
		KeyOptions: []int16{0, 0},
	},
	{
		Name: "vertex_identity_label_graph_id_idx", Relation: "vertex_identity",
		Keys:       []string{"graph_generation_id", "label_generation_id", "graph_id"},
		KeyOptions: []int16{0, 0, 0},
	},
	{
		Name: "edge_identity_lookup_uq", Relation: "edge_identity",
		Unique: true,
		Keys: []string{
			"graph_generation_id", "source_namespace", "label_id", "external_id",
		},
		KeyOptions: []int16{0, 0, 0, 0},
	},
	{
		Name: "edge_identity_graph_id_uq", Relation: "edge_identity",
		Unique:     true,
		Keys:       []string{"graph_generation_id", "graph_id"},
		KeyOptions: []int16{0, 0},
	},
	{
		Name: "edge_identity_label_graph_id_idx", Relation: "edge_identity",
		Keys:       []string{"graph_generation_id", "label_generation_id", "graph_id"},
		KeyOptions: []int16{0, 0, 0},
	},
	{
		Name: "diagnostic_history_recent_idx", Relation: "diagnostic_history",
		Keys:       []string{"recorded_at", "diagnostic_id"},
		KeyOptions: []int16{3, 3},
	},
	{
		Name: "job_label_counter_label_idx", Relation: "job_label_counter",
		Keys:       []string{"label_generation_id", "job_id"},
		KeyOptions: []int16{0, 0},
	},
	{
		Name: "reject_record_attempt_idx", Relation: "reject_record",
		Keys:       []string{"job_id", "batch_id", "attempt"},
		KeyOptions: []int16{0, 0, 0},
	},
	{
		Name: "deferred_edge_resolution_idx", Relation: "deferred_edge",
		Keys: []string{
			"graph_generation_id",
			"start_namespace", "start_label_id", "start_external_id",
			"end_namespace", "end_label_id", "end_external_id",
		},
		KeyOptions: []int16{0, 0, 0, 0, 0, 0, 0},
	},
}

func requiredMetadataIndexesForVersion(version int) []metadataIndexDefinition {
	definitions := make([]metadataIndexDefinition, 0, len(requiredMetadataIndexes))
	for _, definition := range requiredMetadataIndexes {
		if version < 16 && definition.Name == "diagnostic_history_recent_idx" {
			continue
		}
		if version < 17 && (definition.Name == "job_label_counter_label_idx" ||
			strings.HasSuffix(definition.Name, "_graph_id_uq") ||
			strings.HasSuffix(definition.Name, "_label_graph_id_idx")) {
			continue
		}
		definitions = append(definitions, definition)
	}
	return definitions
}

func addMetadataIndexCheck(
	ctx context.Context,
	pool *pgxpool.Pool,
	schema meta.SchemaInspection,
	timeout time.Duration,
	document *report.Document,
) {
	switch schema.State {
	case meta.SchemaAbsent:
		addCheck(document, "metadata-indexes", report.CheckUnavailable,
			"metadata index definitions are unavailable",
			"the metadata schema is absent")
		return
	case meta.SchemaUnknown, meta.SchemaInvalid, meta.SchemaNewer:
		addCheck(document, "metadata-indexes", report.CheckUnknown,
			"metadata index definitions could not be selected safely",
			fmt.Sprintf("metadata schema state=%s", schema.State))
		return
	}
	definitions := requiredMetadataIndexesForVersion(schema.InstalledVersion)
	indexes, err := inspectMetadataIndexes(ctx, pool, timeout, definitions)
	if err != nil {
		addClassifiedCheck(ctx, document, "metadata-indexes",
			"required metadata indexes could not be inspected", err)
		return
	}
	invalid := validateMetadataIndexDefinitions(indexes, definitions)
	status := report.CheckPass
	if len(invalid) > 0 {
		status = report.CheckFail
	}
	addCheck(document, "metadata-indexes", status,
		"required metadata index definitions were inspected",
		fmt.Sprintf(
			"present=%d required=%d invalid=%s",
			len(indexes),
			len(definitions),
			strings.Join(invalid, ","),
		))
}

type metadataIndexInspection struct {
	metadataIndexDefinition
	Valid bool
	Ready bool
}

func inspectMetadataIndexes(
	ctx context.Context,
	pool *pgxpool.Pool,
	timeout time.Duration,
	definitions []metadataIndexDefinition,
) ([]metadataIndexInspection, error) {
	names := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		names = append(names, definition.Name)
	}
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	rows, err := pool.Query(readCtx, `
		SELECT
			index_relation.relname,
			table_relation.relname,
			index_definition.indisvalid,
			index_definition.indisready,
			index_definition.indisunique,
			index_columns.keys,
			index_columns.includes,
			index_columns.key_options,
			COALESCE(
				pg_catalog.pg_get_expr(
					index_definition.indpred,
					index_definition.indrelid,
					true
				),
				''
			)
		FROM pg_catalog.pg_class index_relation
		JOIN pg_catalog.pg_namespace index_namespace
		  ON index_namespace.oid = index_relation.relnamespace
		JOIN pg_catalog.pg_index index_definition
		  ON index_definition.indexrelid = index_relation.oid
		JOIN pg_catalog.pg_class table_relation
		  ON table_relation.oid = index_definition.indrelid
		JOIN pg_catalog.pg_namespace table_namespace
		  ON table_namespace.oid = table_relation.relnamespace
		CROSS JOIN LATERAL (
			SELECT
				COALESCE(
					array_agg(
						COALESCE(attribute.attname::text, '<expression>')
						ORDER BY key.ordinality
					) FILTER (
						WHERE key.ordinality <= index_definition.indnkeyatts
					),
					ARRAY[]::text[]
				) AS keys,
				COALESCE(
					array_agg(
						COALESCE(attribute.attname::text, '<expression>')
						ORDER BY key.ordinality
					) FILTER (
						WHERE key.ordinality > index_definition.indnkeyatts
					),
					ARRAY[]::text[]
				) AS includes,
				COALESCE(
					array_agg(option.value ORDER BY key.ordinality) FILTER (
						WHERE key.ordinality <= index_definition.indnkeyatts
					),
					ARRAY[]::smallint[]
				) AS key_options
			FROM unnest(
				index_definition.indkey::smallint[]
			) WITH ORDINALITY AS key(attnum, ordinality)
			LEFT JOIN pg_catalog.pg_attribute attribute
			  ON attribute.attrelid = table_relation.oid
			 AND attribute.attnum = key.attnum
			LEFT JOIN unnest(
				index_definition.indoption::smallint[]
			) WITH ORDINALITY AS option(value, ordinality)
			  ON option.ordinality = key.ordinality
		) index_columns
		WHERE index_namespace.nspname = 'agefreighter_meta'
		  AND table_namespace.nspname = 'agefreighter_meta'
		  AND index_relation.relname = ANY($1::text[])
		ORDER BY index_relation.relname
		LIMIT $2`,
		names,
		len(definitions)+1,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	indexes := make([]metadataIndexInspection, 0, len(definitions))
	for rows.Next() {
		var index metadataIndexInspection
		if err := rows.Scan(
			&index.Name,
			&index.Relation,
			&index.Valid,
			&index.Ready,
			&index.Unique,
			&index.Keys,
			&index.Include,
			&index.KeyOptions,
			&index.Predicate,
		); err != nil {
			return nil, err
		}
		indexes = append(indexes, index)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return indexes, nil
}

func validateMetadataIndexes(indexes []metadataIndexInspection) []string {
	return validateMetadataIndexDefinitions(indexes, requiredMetadataIndexes)
}

func validateMetadataIndexDefinitions(
	indexes []metadataIndexInspection,
	definitions []metadataIndexDefinition,
) []string {
	observed := make(map[string]metadataIndexInspection, len(indexes))
	for _, index := range indexes {
		observed[index.Name] = index
	}
	invalid := make([]string, 0)
	for _, expected := range definitions {
		actual, ok := observed[expected.Name]
		if !ok ||
			!actual.Valid ||
			!actual.Ready ||
			actual.Relation != expected.Relation ||
			actual.Unique != expected.Unique ||
			!slices.Equal(actual.Keys, expected.Keys) ||
			!slices.Equal(actual.Include, expected.Include) ||
			!slices.Equal(actual.KeyOptions, expected.KeyOptions) ||
			normalizeIndexPredicate(actual.Predicate) !=
				normalizeIndexPredicate(expected.Predicate) {
			invalid = append(invalid, expected.Name)
		}
	}
	return invalid
}

func normalizeIndexPredicate(value string) string {
	value = strings.ToLower(value)
	value = strings.ReplaceAll(value, "::text", "")
	value = strings.ReplaceAll(value, "(", "")
	value = strings.ReplaceAll(value, ")", "")
	return strings.Join(strings.Fields(value), "")
}

func persistDoctor(
	ctx context.Context,
	job config.LoadJob,
	probe age.DegradedProbe,
	inspection meta.SchemaInspection,
	timeout time.Duration,
	document report.Document,
) error {
	encoded, err := report.Render(document, report.FormatJSON)
	if err != nil {
		return fmt.Errorf("render diagnostic report for persistence: %w", err)
	}
	openCtx, cancel := context.WithTimeout(ctx, timeout)
	adapter, store, err := openAGEStore(openCtx, job)
	cancel()
	if err != nil {
		return fmt.Errorf("persist doctor report: open current target: %w", err)
	}
	defer adapter.Close()
	checkCtx, checkCancel := context.WithTimeout(ctx, timeout)
	current, err := store.InspectSchema(checkCtx)
	checkCancel()
	if err != nil {
		return fmt.Errorf("persist doctor report: inspect metadata schema: %w", err)
	}
	if err := current.RequireCurrent(); err != nil ||
		current.InstalledVersion != inspection.InstalledVersion {
		if err == nil {
			err = errors.New("metadata schema changed during diagnosis")
		}
		return fmt.Errorf("persist doctor report: %w", err)
	}
	writeCtx, writeCancel := context.WithTimeout(ctx, timeout)
	_, err = store.PersistDiagnostic(writeCtx, meta.DiagnosticRecord{
		Outcome:                 string(document.Outcome),
		TargetGraph:             job.Target.Graph,
		PostgreSQLVersionNumber: probe.PostgreSQLVersionNumber,
		AGEVersion:              boundedTypedValue(probe.AGEVersion, 64),
		MetadataSchemaVersion:   inspection.InstalledVersion,
		Report:                  encoded,
	})
	writeCancel()
	if err != nil {
		return fmt.Errorf("persist doctor report: %w", err)
	}
	return nil
}

func addUnavailableAGEChecks(document *report.Document) {
	for _, value := range []struct{ id, summary string }{
		{"age-search-path", "AGE search path is unavailable"},
		{"target-graph", "AGE target graph catalog is unavailable"},
		{"graph-catalog-objects", "AGE label, relation, and sequence catalogs are unavailable"},
	} {
		addCheck(document, value.id, report.CheckUnavailable,
			value.summary, "prerequisite Apache AGE capability did not pass")
	}
}

func addUnavailableMetadataChecks(document *report.Document) {
	for _, value := range []struct{ id, summary string }{
		{"metadata-jobs", "load-job health is unavailable"},
		{"graph-generations", "graph-generation health is unavailable"},
		{"retained-backups", "retained-backup health is unavailable"},
		{"catalog-residue", "residue and orphan checks are unavailable"},
	} {
		addCheck(document, value.id, report.CheckUnavailable,
			value.summary, "a read-compatible metadata schema is required")
	}
}

func queryDoctorRow(
	ctx context.Context,
	pool *pgxpool.Pool,
	timeout time.Duration,
	statement string,
	arguments ...any,
) pgx.Row {
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	return cancelRow{
		Row:    pool.QueryRow(readCtx, statement, arguments...),
		cancel: cancel,
	}
}

type cancelRow struct {
	pgx.Row
	cancel context.CancelFunc
}

func (row cancelRow) Scan(dest ...any) error {
	defer row.cancel()
	return row.Row.Scan(dest...)
}

func addCatalogResidueCheck(
	ctx context.Context,
	pool *pgxpool.Pool,
	timeout time.Duration,
	document *report.Document,
) {
	var inspected, missing, mismatched int
	err := queryDoctorRow(ctx, pool, timeout, `
		SELECT
			COUNT(*)::integer,
			COUNT(*) FILTER (WHERE a.graphid IS NULL)::integer,
			COUNT(*) FILTER (
				WHERE a.graphid IS NOT NULL
				  AND (a.name::text <> bounded.graph_name
				       OR a.namespace::oid <> bounded.namespace_oid)
			)::integer
		FROM (
			SELECT graph_name, graph_oid, namespace_oid
			FROM agefreighter_meta.graph_generation
			WHERE state IN ('loading', 'active', 'retired')
			ORDER BY graph_generation_id DESC
			LIMIT $1
		) bounded
		LEFT JOIN ag_catalog.ag_graph a ON a.graphid = bounded.graph_oid`,
		MaxDoctorCatalogEntries+1,
	).Scan(&inspected, &missing, &mismatched)
	if err != nil {
		addClassifiedCheck(ctx, document, "catalog-residue",
			"bounded residue and orphan indicators could not be inspected", err)
		return
	}
	status := report.CheckPass
	if missing > 0 || mismatched > 0 {
		status = report.CheckWarning
	}
	if inspected > MaxDoctorCatalogEntries {
		status = report.CheckUnknown
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "RESIDUE_CHECK_TRUNCATED",
			Message: fmt.Sprintf(
				"residue checks are limited to %d newest graph generations",
				MaxDoctorCatalogEntries,
			),
		})
	}
	addCheck(document, "catalog-residue", status,
		"bounded metadata-to-AGE catalog residue indicators were inspected",
		fmt.Sprintf(
			"inspected=%d missing_graph_catalog=%d mismatched_graph_catalog=%d limit=%d",
			inspected,
			missing,
			mismatched,
			MaxDoctorCatalogEntries,
		))
}

func boundedTypedValue(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit]
}

func addValueCheck(
	ctx context.Context,
	document *report.Document,
	id string,
	err error,
	summary, detail string,
) {
	if err != nil {
		addClassifiedCheck(ctx, document, id, summary+" could not be read", err)
		return
	}
	addCheck(document, id, report.CheckPass, summary+" was inspected", detail)
}

func addClassifiedCheck(
	ctx context.Context,
	document *report.Document,
	id, summary string,
	err error,
) {
	if ctx.Err() != nil {
		addCheck(document, id, report.CheckUnknown, summary,
			"diagnostic operation was canceled")
		return
	}
	addCheck(document, id, diagnosticErrorStatus(err), summary,
		safeDatabaseDetail(err, "database catalog operation failed"))
}

func addCheck(
	document *report.Document,
	id string,
	status report.CheckStatus,
	summary, detail string,
) {
	document.Checks = append(document.Checks, report.Check{
		ID: id, Status: status, Summary: summary,
		Detail: boundedReportValue(detail),
	})
}

func probeCheckStatus(status age.ProbeStatus) report.CheckStatus {
	switch status {
	case age.ProbePass:
		return report.CheckPass
	case age.ProbeFail:
		return report.CheckFail
	case age.ProbeUnavailable:
		return report.CheckUnavailable
	default:
		return report.CheckUnknown
	}
}

func diagnosticErrorStatus(err error) report.CheckStatus {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch pgErr.Code {
		case "42501":
			return report.CheckUnknown
		case "42P01", "3F000", "42883":
			return report.CheckUnavailable
		}
	}
	return report.CheckUnknown
}

func catalogErrorStatus(err error) report.CheckStatus {
	if errors.Is(err, age.ErrCatalogEntryNotFound) ||
		strings.Contains(err.Error(), "catalog mismatch") ||
		strings.Contains(err.Error(), "invalid kind") ||
		strings.Contains(err.Error(), "invalid ID") {
		return report.CheckFail
	}
	return diagnosticErrorStatus(err)
}

func safeDatabaseDetail(err error, fallback string) string {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch pgErr.Code {
		case "42501":
			return "permission denied; grant only the catalog visibility required for this check"
		case "42P01", "3F000", "42883":
			return "required catalog object is unavailable"
		case "57014":
			return "diagnostic operation exceeded its configured deadline"
		default:
			return fmt.Sprintf("%s (SQLSTATE %s)", fallback, pgErr.Code)
		}
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "diagnostic operation exceeded its configured deadline"
	}
	if errors.Is(err, context.Canceled) {
		return "diagnostic operation was canceled"
	}
	return fallback
}

func canceled(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	return nil
}

func searchPathContains(value, expected string) bool {
	for part := range strings.SplitSeq(value, ",") {
		if strings.Trim(strings.TrimSpace(part), `"`) == expected {
			return true
		}
	}
	return false
}

func finalizeDoctor(document *report.Document) {
	document.Outcome = report.OutcomePass
	if hasStatus(*document, report.CheckFail) || len(document.Errors) > 0 {
		document.Outcome = report.OutcomeFail
		return
	}
	if hasStatus(*document, report.CheckUnknown) ||
		hasStatus(*document, report.CheckUnavailable) ||
		len(document.IncompleteChecks) > 0 {
		document.Outcome = report.OutcomeIncomplete
	}
}
