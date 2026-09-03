package app

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

const (
	MaxReportBatches       = 100
	maxReportLabels        = 32
	maxReportRejectRecords = 100
	reportValueBytes       = 1024
)

type ReportOptions struct {
	IncludeCounts bool
	LimitBatches  int
	GeneratedAt   time.Time
}

func MigrationReport(
	ctx context.Context,
	path string,
	jobID string,
	options ReportOptions,
) (report.Document, error) {
	if err := meta.ValidateJobID(jobID); err != nil {
		return report.Document{}, err
	}
	if options.LimitBatches < 0 || options.LimitBatches > MaxReportBatches {
		return report.Document{}, fmt.Errorf(
			"batch report limit must be within 0..%d",
			MaxReportBatches,
		)
	}
	jobConfig, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load target configuration: %w", err)
	}
	timeout := time.Duration(jobConfig.Runtime.OperationTimeout)
	openCtx, cancel := context.WithTimeout(ctx, timeout)
	probe, err := probeTarget(openCtx, jobConfig)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	openCtx, cancel = context.WithTimeout(ctx, timeout)
	target, err := openReadOnlyTarget(openCtx, jobConfig)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	if err := target.Metadata.RequireReadCompatible(); err != nil {
		return report.Document{}, err
	}

	readCtx, cancel := context.WithTimeout(ctx, timeout)
	storedJob, err := target.Store.GetJob(readCtx, jobID)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	if err := validateStoredTargetIdentity(jobConfig, storedJob); err != nil {
		return report.Document{}, err
	}

	snapshot := migrationReportSnapshot{
		Probe:      probe,
		Schema:     target.Metadata,
		Job:        storedJob,
		Counts:     make(map[int64]countResult),
		BatchLimit: options.LimitBatches,
	}
	readCtx, cancel = context.WithTimeout(ctx, timeout)
	snapshot.Graph, err = target.Store.GraphGenerationForJob(readCtx, jobID)
	cancel()
	if err != nil && !errors.Is(err, meta.ErrNotFound) {
		return report.Document{}, err
	}
	snapshot.GraphAvailable = err == nil

	if snapshot.GraphAvailable {
		readCtx, cancel = context.WithTimeout(ctx, timeout)
		labels, labelErr := target.Store.ListLabelGenerations(
			readCtx,
			snapshot.Graph.ID,
			maxReportLabels+1,
		)
		cancel()
		if labelErr != nil {
			return report.Document{}, labelErr
		}
		if len(labels) > maxReportLabels {
			snapshot.LabelsTruncated = true
			labels = labels[:maxReportLabels]
		}
		snapshot.Labels = labels
	}

	readCtx, cancel = context.WithTimeout(ctx, timeout)
	snapshot.LatestBatch, err = target.Store.LatestBatch(readCtx, jobID)
	cancel()
	if err != nil && !errors.Is(err, meta.ErrNotFound) {
		return report.Document{}, err
	}
	snapshot.LatestBatchAvailable = err == nil

	if options.LimitBatches > 0 {
		readCtx, cancel = context.WithTimeout(ctx, timeout)
		batches, batchErr := target.Store.ListBatches(
			readCtx,
			jobID,
			options.LimitBatches+1,
		)
		cancel()
		if batchErr != nil {
			return report.Document{}, batchErr
		}
		if len(batches) > options.LimitBatches {
			snapshot.BatchesTruncated = true
			batches = batches[:options.LimitBatches]
		}
		snapshot.Batches = batches
	}

	readCtx, cancel = context.WithTimeout(ctx, timeout)
	snapshot.Rejects, err = target.Store.ListRejectSummaries(
		readCtx,
		jobID,
		maxReportRejectRecords,
	)
	cancel()
	if err != nil {
		return report.Document{}, err
	}

	if target.Metadata.InstalledVersion >= 15 {
		readCtx, cancel = context.WithTimeout(ctx, timeout)
		snapshot.Telemetry, err = target.Store.GetConnectorTelemetry(readCtx, jobID)
		cancel()
		if err != nil && !errors.Is(err, meta.ErrNotFound) {
			return report.Document{}, err
		}
		snapshot.TelemetryAvailable = err == nil
	}

	if options.IncludeCounts {
		for _, label := range snapshot.Labels {
			countCtx, countCancel := context.WithTimeout(ctx, timeout)
			count, countErr := target.Store.CountLabelIdentitiesWithTimeout(
				countCtx,
				snapshot.Graph.ID,
				label.ID,
				label.Kind,
				timeout,
			)
			countCancel()
			if err := ctx.Err(); err != nil {
				return report.Document{}, err
			}
			snapshot.Counts[label.ID] = classifyCount(count, countErr)
		}
	}
	generatedAt := options.GeneratedAt
	if generatedAt.IsZero() {
		generatedAt = time.Now()
	}
	return buildMigrationReport(snapshot, options.IncludeCounts, generatedAt)
}

type migrationReportSnapshot struct {
	Probe                age.DegradedProbe
	Schema               meta.SchemaInspection
	Job                  meta.Job
	Graph                meta.GraphGeneration
	GraphAvailable       bool
	Labels               []meta.LabelGeneration
	LabelsTruncated      bool
	LatestBatch          meta.BatchAttempt
	LatestBatchAvailable bool
	Batches              []meta.BatchAttempt
	BatchLimit           int
	BatchesTruncated     bool
	Rejects              meta.RejectSummaryPage
	Telemetry            meta.ConnectorTelemetry
	TelemetryAvailable   bool
	Counts               map[int64]countResult
}

type countResult struct {
	Count  int64
	Status report.CheckStatus
	Detail string
}

func classifyCount(count int64, err error) countResult {
	if err == nil {
		return countResult{Count: count, Status: report.CheckPass}
	}
	result := countResult{Status: report.CheckFail, Detail: boundedReportValue(err.Error())}
	var pgErr *pgconn.PgError
	switch {
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		result.Status = report.CheckUnknown
		result.Detail = "identity count did not complete before its operation deadline"
	case errors.As(err, &pgErr) && pgErr.Code == "57014":
		result.Status = report.CheckUnknown
		result.Detail = "identity count was canceled by the statement timeout"
	case errors.As(err, &pgErr) && pgErr.Code == "42501":
		result.Status = report.CheckUnknown
		result.Detail = "permission denied while counting identity rows"
	}
	return result
}

func buildMigrationReport(
	snapshot migrationReportSnapshot,
	includeCounts bool,
	generatedAt time.Time,
) (report.Document, error) {
	document := report.New("report", generatedAt)
	document.Job = &report.Job{
		ID:                snapshot.Job.ID,
		ConfigFingerprint: snapshot.Job.ConfigFingerprint,
	}
	document.Target = &report.Target{
		PostgreSQL: versionValue(
			snapshot.Probe.PostgreSQLVersion,
			snapshot.Probe.PostgreSQLStatus,
		),
		AGE: versionValue(
			snapshot.Probe.AGEVersion,
			snapshot.Probe.AGEVersionStatus,
		),
	}
	document.Checks = append(document.Checks,
		jobStatusCheck(snapshot.Job),
		schemaCheck(snapshot.Schema),
	)
	document.Sections = append(document.Sections, jobSection(snapshot.Job))
	document.Sections = append(
		document.Sections,
		schemaSection(snapshot.Schema),
		batchSection(snapshot),
		rejectSection(snapshot.Rejects),
		telemetrySection(snapshot),
	)
	if snapshot.GraphAvailable {
		document.Sections = append(document.Sections, graphSection(snapshot.Graph))
	} else {
		document.Sections = append(document.Sections, unavailableSection(
			"Graph generation",
			"generation",
			"no graph generation is recorded for this job",
		))
	}
	document.Sections = append(document.Sections, labelSection(snapshot, includeCounts))
	document.Sections = append(document.Sections, backupSection(snapshot.Job, snapshot.Graph))

	if snapshot.LabelsTruncated {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "LABELS_TRUNCATED",
			Message: fmt.Sprintf(
				"label generation output is limited to %d entries",
				maxReportLabels,
			),
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "label-generations")
	}
	if snapshot.BatchesTruncated {
		document.Warnings = append(document.Warnings, report.Finding{
			Code:    "BATCHES_TRUNCATED",
			Message: "additional batch attempts exist beyond --limit-batches",
		})
	}
	if snapshot.Rejects.Truncated {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "REJECTS_TRUNCATED",
			Message: fmt.Sprintf(
				"reject summary counts cover only the first %d persisted rejects",
				snapshot.Rejects.ScannedRows,
			),
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "reject-summary")
	}
	for _, label := range snapshot.Labels {
		result, exists := snapshot.Counts[label.ID]
		if !exists {
			continue
		}
		if result.Status == report.CheckUnknown {
			document.IncompleteChecks = append(document.IncompleteChecks, "identity-counts")
			break
		}
		if result.Status == report.CheckFail {
			document.Errors = append(document.Errors, report.Finding{
				Code:    "IDENTITY_COUNT_ERROR",
				Message: result.Detail,
			})
			break
		}
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
		return report.Document{}, err
	}
	return document, nil
}

func jobStatusCheck(job meta.Job) report.Check {
	check := report.Check{
		ID:      "job-status",
		Summary: fmt.Sprintf("load job is %s", job.Status),
	}
	switch job.Status {
	case meta.JobCommitted:
		check.Status = report.CheckPass
	case meta.JobFailed:
		check.Status = report.CheckFail
		check.Detail = "the job has a persisted failure; inspect protected operational logs"
	default:
		check.Status = report.CheckWarning
	}
	return check
}

func schemaCheck(schema meta.SchemaInspection) report.Check {
	check := report.Check{
		ID:      "metadata-schema",
		Summary: fmt.Sprintf("metadata schema is %s", schema.State),
		Detail:  boundedReportValue(schema.Detail),
	}
	if schema.State == meta.SchemaCurrent {
		check.Status = report.CheckPass
	} else {
		check.Status = report.CheckWarning
		if check.Detail == "" {
			check.Detail = fmt.Sprintf(
				"schema v%d is read-compatible; load or resume upgrades to v%d",
				schema.InstalledVersion,
				schema.SupportedVersion,
			)
		}
	}
	return check
}

func jobSection(job meta.Job) report.Section {
	fields := []report.Field{
		passField("committedBytes", strconv.FormatInt(job.CommittedBytes, 10)),
		passField("committedRows", strconv.FormatInt(job.CommittedRows, 10)),
		passField("createdAt", formatTime(job.CreatedAt)),
		passField("loadMode", job.LoadMode),
		passField("name", job.Name),
		passField("nextBatchId", strconv.FormatUint(job.NextBatchID, 10)),
		passField("rejectedRows", strconv.FormatInt(job.RejectedRows, 10)),
		passField("sourceRejectedRows", strconv.FormatInt(job.SourceRejectedRows, 10)),
		passField("sourceType", job.SourceType),
		passField("status", string(job.Status)),
		passField("targetBackend", string(job.TargetBackend)),
		passField("targetGraph", job.TargetGraph),
		passField("targetSchema", valueOrNone(job.TargetSchema)),
		passField("updatedAt", formatTime(job.UpdatedAt)),
	}
	fields = append(fields, optionalTimeField("startedAt", job.StartedAt))
	fields = append(fields, optionalTimeField("completedAt", job.CompletedAt))
	return report.Section{Title: "Job", Fields: fields}
}

func schemaSection(schema meta.SchemaInspection) report.Section {
	return report.Section{Title: "Metadata schema", Fields: []report.Field{
		passField("installedVersion", strconv.Itoa(schema.InstalledVersion)),
		passField("pendingVersions", strconv.Itoa(schema.PendingVersions)),
		passField("state", string(schema.State)),
		passField("supportedVersion", strconv.Itoa(schema.SupportedVersion)),
	}}
}

func graphSection(graph meta.GraphGeneration) report.Section {
	return report.Section{Title: "Graph generation", Fields: []report.Field{
		passField("generation", strconv.FormatUint(graph.Generation, 10)),
		passField("graphGenerationId", strconv.FormatInt(graph.ID, 10)),
		passField("graphName", graph.GraphName),
		passField("graphOid", strconv.FormatUint(uint64(graph.GraphOID), 10)),
		passField("namespaceOid", strconv.FormatUint(uint64(graph.NamespaceOID), 10)),
		passField("replacesGraphOid", strconv.FormatUint(uint64(graph.ReplacesGraphOID), 10)),
		passField("state", string(graph.State)),
	}}
}

func labelSection(snapshot migrationReportSnapshot, includeCounts bool) report.Section {
	section := report.Section{Title: "Label generations", Fields: []report.Field{}}
	if !snapshot.GraphAvailable {
		section.Fields = append(section.Fields, unavailableField(
			"labels",
			"graph generation is unavailable",
		))
		return section
	}
	if len(snapshot.Labels) == 0 {
		section.Fields = append(section.Fields, passField("labels", "none"))
		return section
	}
	for index, label := range snapshot.Labels {
		prefix := fmt.Sprintf("%03d.%s", index+1, label.LabelName)
		value := fmt.Sprintf(
			"kind=%s,id=%d,labelId=%d,relationOid=%d,sequenceOid=%d,mappingGeneration=%d",
			string(byte(label.Kind)),
			label.ID,
			label.LabelID,
			label.RelationOID,
			label.SequenceOID,
			label.MappingGeneration,
		)
		section.Fields = append(section.Fields, passField(prefix+".identity", value))
		if !includeCounts {
			continue
		}
		countName := prefix + ".exactIdentityCount"
		result, exists := snapshot.Counts[label.ID]
		if !exists {
			section.Fields = append(section.Fields, unavailableField(
				countName,
				"identity count result is unavailable",
			))
			continue
		}
		field := report.Field{Name: countName, Status: result.Status}
		if result.Status == report.CheckPass {
			field.Value = strconv.FormatInt(result.Count, 10)
		} else {
			field.Value = result.Detail
		}
		section.Fields = append(section.Fields, field)
	}
	return section
}

func batchSection(snapshot migrationReportSnapshot) report.Section {
	section := report.Section{Title: "Batch attempts", Fields: []report.Field{}}
	if snapshot.LatestBatchAvailable {
		latest := snapshot.LatestBatch
		section.Fields = append(section.Fields,
			passField("latest.batchId", strconv.FormatUint(latest.BatchID, 10)),
			passField("latest.attempt", strconv.FormatUint(uint64(latest.Attempt), 10)),
			passField("latest.status", string(latest.Status)),
			passField("latest.lastResource", boundedReportValue(latest.Last.Resource)),
			passField("latest.lastLine", strconv.FormatInt(latest.Last.Line, 10)),
			passField("latest.lastByteOffset", strconv.FormatInt(latest.Last.ByteOffset, 10)),
			passField("latest.lastTokenDigest", tokenDigest(latest.Last.Token)),
		)
	} else {
		section.Fields = append(section.Fields, passField("latest", "none"))
	}
	if snapshot.BatchLimit == 0 {
		return section
	}
	for _, batch := range snapshot.Batches {
		name := fmt.Sprintf("batch.%020d.attempt.%010d", batch.BatchID, batch.Attempt)
		value := fmt.Sprintf(
			"status=%s,rows=%d,bytes=%d,rejectedRows=%d,lastLine=%d,lastByteOffset=%d,lastTokenDigest=%s",
			batch.Status,
			batch.Rows,
			batch.Bytes,
			batch.RejectedRows,
			batch.Last.Line,
			batch.Last.ByteOffset,
			tokenDigest(batch.Last.Token),
		)
		section.Fields = append(section.Fields, passField(name, value))
	}
	return section
}

func rejectSection(page meta.RejectSummaryPage) report.Section {
	section := report.Section{Title: "Reject summary", Fields: []report.Field{
		passField("scannedRows", strconv.Itoa(page.ScannedRows)),
		passField("truncated", strconv.FormatBool(page.Truncated)),
	}}
	if len(page.Summaries) == 0 {
		section.Fields = append(section.Fields, passField("classes", "none in bounded window"))
		return section
	}
	for index, summary := range page.Summaries {
		section.Fields = append(section.Fields, passField(
			fmt.Sprintf("class.%03d", index+1),
			fmt.Sprintf("%s=%d", boundedReportValue(summary.ErrorClass), summary.Count),
		))
	}
	return section
}

func telemetrySection(snapshot migrationReportSnapshot) report.Section {
	if !snapshot.TelemetryAvailable {
		detail := "connector telemetry was not recorded for this job"
		if snapshot.Schema.InstalledVersion < 15 {
			detail = "connector telemetry requires metadata schema v15 and is unavailable for v14 jobs"
		}
		return unavailableSection("Connector telemetry", "summary", detail)
	}
	value := snapshot.Telemetry
	return report.Section{Title: "Connector telemetry", Fields: []report.Field{
		passField("connector", value.Connector),
		passField("continuationDigest", value.ContinuationDigest),
		passField("failedRequestAttempts", strconv.FormatInt(value.FailedRequestAttempts, 10)),
		passField("pages", strconv.FormatInt(value.Pages, 10)),
		passField("recordedAt", formatTime(value.RecordedAt)),
		passField("requestCharge", strconv.FormatFloat(value.RequestCharge, 'f', -1, 64)),
		passField("throttledRequests", strconv.FormatInt(value.ThrottledRequests, 10)),
	}}
}

func backupSection(job meta.Job, graph meta.GraphGeneration) report.Section {
	fields := []report.Field{
		passField("backupGraphName", valueOrNone(job.BackupGraphName)),
		passField("retained", strconv.FormatBool(
			job.BackupGraphName != "" && job.BackupCleanedAt == nil,
		)),
	}
	if graph.ReplacesGraphOID != 0 {
		fields = append(fields, passField(
			"backupGraphOid",
			strconv.FormatUint(uint64(graph.ReplacesGraphOID), 10),
		))
	}
	if job.BackupCleanedAt != nil {
		fields = append(fields, passField("cleanedAt", formatTime(*job.BackupCleanedAt)))
	}
	return report.Section{Title: "Replacement backup", Fields: fields}
}

func valueOrNone(value string) string {
	if value == "" {
		return "none"
	}
	return value
}

func unavailableSection(title, name, detail string) report.Section {
	return report.Section{
		Title:  title,
		Fields: []report.Field{unavailableField(name, detail)},
	}
}

func unavailableField(name, detail string) report.Field {
	return report.Field{
		Name:   name,
		Value:  boundedReportValue(detail),
		Status: report.CheckUnavailable,
	}
}

func passField(name, value string) report.Field {
	return report.Field{
		Name:   name,
		Value:  boundedReportValue(value),
		Status: report.CheckPass,
	}
}

func optionalTimeField(name string, value *time.Time) report.Field {
	if value == nil {
		return unavailableField(name, "not recorded")
	}
	return passField(name, formatTime(*value))
}

func formatTime(value time.Time) string {
	return value.UTC().Format(time.RFC3339Nano)
}

func tokenDigest(value string) string {
	if value == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func boundedReportValue(value string) string {
	value = strings.ReplaceAll(value, "\x00", "\uFFFD")
	if len(value) <= reportValueBytes {
		return value
	}
	value = value[:reportValueBytes-3]
	for !utf8.ValidString(value) {
		value = value[:len(value)-1]
	}
	return value + "..."
}

func versionValue(value string, status age.ProbeStatus) report.VersionValue {
	result := report.VersionValue{Value: value}
	switch status {
	case age.ProbePass:
		result.Status = report.CheckPass
	case age.ProbeFail:
		result.Status = report.CheckFail
	case age.ProbeUnavailable:
		result.Status = report.CheckUnavailable
	default:
		result.Status = report.CheckUnknown
	}
	return result
}

func hasStatus(document report.Document, expected report.CheckStatus) bool {
	if document.Target != nil &&
		(document.Target.PostgreSQL.Status == expected ||
			document.Target.AGE.Status == expected) {
		return true
	}
	for _, check := range document.Checks {
		if check.Status == expected {
			return true
		}
	}
	for _, section := range document.Sections {
		for _, field := range section.Fields {
			if field.Status == expected {
				return true
			}
		}
	}
	return false
}
