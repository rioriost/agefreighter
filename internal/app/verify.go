package app

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

const (
	DefaultIntegrityLimit = 100
	MaxIntegrityLimit     = 1000
	maxVerifyLabels       = 128
	maxResolvedMapLabels  = 1024
)

type VerifyOptions struct {
	Counts      bool
	Integrity   bool
	Limit       int
	GeneratedAt time.Time
}

func VerificationReport(
	ctx context.Context,
	path string,
	jobID string,
	options VerifyOptions,
) (report.Document, error) {
	if err := meta.ValidateJobID(jobID); err != nil {
		return report.Document{}, err
	}
	if !options.Counts && !options.Integrity {
		return report.Document{}, errors.New("deep verification requires counts or integrity")
	}
	if options.Limit == 0 {
		options.Limit = DefaultIntegrityLimit
	}
	if options.Limit < 1 || options.Limit > MaxIntegrityLimit {
		return report.Document{}, fmt.Errorf(
			"integrity limit must be within 1..%d", MaxIntegrityLimit,
		)
	}
	jobConfig, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load target configuration: %w", err)
	}
	if jobConfig.Target.Type == config.TargetPostgreSQLPropertyGraph {
		return propertyGraphVerificationReport(ctx, jobConfig, jobID, options)
	}
	timeout := time.Duration(jobConfig.Runtime.OperationTimeout)
	openCtx, cancel := context.WithTimeout(ctx, timeout)
	target, err := openReadOnlyTarget(openCtx, jobConfig)
	cancel()
	if err != nil {
		return report.Document{}, err
	}
	defer target.Runtime.Close()
	adapter, err := target.AGEAdapter()
	if err != nil {
		return report.Document{}, err
	}
	if err := target.Metadata.RequireReadCompatible(); err != nil {
		return report.Document{}, err
	}
	readCtx, readCancel := context.WithTimeout(ctx, timeout)
	stored, err := target.Store.GetJob(readCtx, jobID)
	readCancel()
	if err != nil {
		return report.Document{}, err
	}
	if err := validateStoredTargetIdentity(jobConfig, stored); err != nil {
		return report.Document{}, err
	}
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	document := report.New("verify", at)
	document.Job = &report.Job{
		ID: stored.ID, ConfigFingerprint: stored.ConfigFingerprint,
	}
	status := report.CheckFail
	if stored.Status == meta.JobCommitted {
		status = report.CheckPass
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "job-status", Status: status,
		Summary: fmt.Sprintf("load job is %s", stored.Status),
	})
	if stored.Status != meta.JobCommitted {
		document.Outcome = report.OutcomeFail
		return validatedVerificationReport(document)
	}

	expectedLabels, identityCoverageByLabel, mappingAvailable := addVerificationMetadataCheck(
		ctx, target.Store, target.Metadata, jobConfig, stored, timeout, &document,
	)
	readCtx, readCancel = context.WithTimeout(ctx, timeout)
	graph, err := target.Store.GraphGenerationForJob(readCtx, jobID)
	readCancel()
	if err != nil {
		return report.Document{}, err
	}
	graphStatus := report.CheckPass
	graphDetail := "active graph generation matches the configured target"
	if graph.State != meta.GenerationActive || graph.GraphName != jobConfig.Target.Graph {
		graphStatus = report.CheckFail
		graphDetail = "committed graph generation is not active at the configured target"
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "graph-generation", Status: graphStatus,
		Summary: "graph generation ownership was checked", Detail: graphDetail,
	})
	labels := []meta.LabelGeneration{}
	if mappingAvailable && len(expectedLabels) > maxVerifyLabels {
		addVerificationLabelLimit(&document, options, len(expectedLabels))
		document.Outcome = ""
		return validatedVerificationReport(document)
	}
	if mappingAvailable {
		ids := make([]int64, len(expectedLabels))
		for index, label := range expectedLabels {
			ids[index] = label.ID
		}
		readCtx, readCancel = context.WithTimeout(ctx, timeout)
		actualLabels, labelErr := target.Store.ListLabelGenerationsByID(
			readCtx, graph.ID, ids,
		)
		readCancel()
		if labelErr != nil {
			return report.Document{}, labelErr
		}
		var mismatches []string
		labels, mismatches = reconcileExpectedLabels(expectedLabels, actualLabels)
		labelStatus := report.CheckPass
		labelDetail := fmt.Sprintf("%d expected label generations matched", len(labels))
		if len(mismatches) > 0 {
			labelStatus = report.CheckFail
			labelDetail = "missing or changed expected label generations: " +
				strings.Join(mismatches, ",")
		}
		document.Checks = append(document.Checks, report.Check{
			ID: "resolved-label-generations", Status: labelStatus,
			Summary: "persisted resolved label generations were checked",
			Detail:  boundedReportValue(labelDetail),
		})
	} else {
		document.IncompleteChecks = append(document.IncompleteChecks, "label-generations")
	}
	ownershipStatus := report.CheckPass
	ownershipDetail := "label generations belong to the active graph generation"
	if !mappingAvailable {
		ownershipStatus = report.CheckUnavailable
		ownershipDetail = "persisted expected label generations are unavailable"
	}
	for _, label := range labels {
		if label.GraphGenerationID != graph.ID ||
			label.GraphNamespaceOID != graph.NamespaceOID {
			ownershipStatus = report.CheckFail
			ownershipDetail = "a label generation is not owned by the active graph generation"
			break
		}
	}
	if ownershipStatus == report.CheckPass &&
		stored.BackupGraphName != "" && stored.BackupCleanedAt == nil &&
		(graph.ReplacesGraphOID == 0 || graph.ReplacesGraphOID == graph.GraphOID) {
		ownershipStatus = report.CheckFail
		ownershipDetail = "retained replacement backup is not isolated from the active generation"
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "generation-ownership", Status: ownershipStatus,
		Summary: "generation ownership and retained-backup isolation were checked",
		Detail:  ownershipDetail,
	})
	counters := map[int64]meta.LabelCounter{}
	if options.Counts && target.Metadata.InstalledVersion >= 17 {
		ids := make([]int64, len(labels))
		for index, label := range labels {
			ids[index] = label.ID
		}
		readCtx, readCancel = context.WithTimeout(ctx, timeout)
		values, counterErr := target.Store.ListLabelCountersByID(
			readCtx, jobID, ids,
		)
		readCancel()
		if counterErr != nil {
			if err := ctx.Err(); err != nil {
				return report.Document{}, err
			}
			document.Checks = append(document.Checks, classifiedVerificationCheck(
				"stored-label-counters", "stored per-label counters could not be read", counterErr,
			))
		} else {
			for _, value := range values {
				counters[value.LabelGenerationID] = value
			}
		}
	}
	countFields := []report.Field{}
	integrityFields := []report.Field{}
	if !mappingAvailable {
		if options.Counts {
			countFields = append(countFields, report.Field{
				Name: "expected-labels", Status: report.CheckUnavailable,
				Value: "counts were not run without a validated persisted label set",
			})
		}
		if options.Integrity {
			integrityFields = append(integrityFields, report.Field{
				Name: "expected-labels", Status: report.CheckUnavailable,
				Value: "integrity was not run without a validated persisted label set",
			})
		}
	}
	for _, label := range labels {
		coverage := identityCoverageByLabel[label.ID]
		var countResult liveCountResult
		if options.Counts {
			catalogResult, liveResult := verifyLiveLabel(
				ctx, target.Store, adapter, graph, label, coverage, timeout,
			)
			document.Checks = append(document.Checks, catalogResult)
			countResult = liveResult
			if err := ctx.Err(); err != nil {
				return report.Document{}, err
			}
			field := countVerificationField(
				label, coverage, countResult, counters,
				target.Metadata.InstalledVersion,
			)
			countFields = append(countFields, field)
			if field.Status == report.CheckUnknown || field.Status == report.CheckUnavailable {
				document.IncompleteChecks = append(
					document.IncompleteChecks, "counts."+label.LabelName,
				)
			} else if label.Kind == meta.EdgeLabel &&
				coverage != identityCoverageFull {
				document.IncompleteChecks = append(
					document.IncompleteChecks, "counts."+label.LabelName+".identity-coverage",
				)
			}
		}
		if options.Integrity {
			field, truncated := integrityVerificationField(
				ctx, adapter, graph, label, coverage,
				options.Limit, timeout,
			)
			if err := ctx.Err(); err != nil {
				return report.Document{}, err
			}
			integrityFields = append(integrityFields, field)
			if truncated || field.Status == report.CheckUnknown ||
				field.Status == report.CheckUnavailable {
				document.IncompleteChecks = append(
					document.IncompleteChecks, "integrity."+label.LabelName,
				)
			}
		}
	}
	if options.Counts {
		countFields = append(countFields, unclassifiedRejectField(
			ctx, target.Store, target.Metadata.InstalledVersion, jobID, timeout,
		))
		document.Sections = append(document.Sections, report.Section{
			Title: "Per-label counts", Fields: countFields,
		})
	}
	if options.Integrity {
		document.Sections = append(document.Sections, report.Section{
			Title: "Bounded integrity", Fields: integrityFields,
		})
	}
	document.Outcome = report.OutcomePass
	if hasStatus(document, report.CheckFail) {
		document.Outcome = report.OutcomeFail
	} else if hasStatus(document, report.CheckUnknown) ||
		hasStatus(document, report.CheckUnavailable) ||
		len(document.IncompleteChecks) > 0 {
		document.Outcome = report.OutcomeIncomplete
	}
	return validatedVerificationReport(document)
}

type liveCountResult struct {
	IdentityRows int64
	PhysicalRows int64
	Status       report.CheckStatus
	Detail       string
}

func verifyLiveLabel(
	ctx context.Context,
	store *meta.Store,
	adapter *age.Adapter,
	graph meta.GraphGeneration,
	label meta.LabelGeneration,
	coverage identityCoverage,
	timeout time.Duration,
) (report.Check, liveCountResult) {
	countCtx, cancel := context.WithTimeout(ctx, timeout)
	count, countErr := store.CountLabelIdentitiesWithTimeout(
		countCtx, graph.ID, label.ID, label.Kind, timeout,
	)
	cancel()
	id := "catalog." + string(byte(label.Kind)) + "." + label.LabelName
	if countErr != nil {
		check := classifiedVerificationCheck(id, "catalog verification could not count identities", countErr)
		return check, liveCountResult{
			Status: check.Status, Detail: check.Detail,
		}
	}
	live := liveCountResult{IdentityRows: count}
	checkCtx, checkCancel := context.WithTimeout(ctx, timeout)
	err := adapter.InTransaction(checkCtx, func(transaction *age.Transaction) error {
		if err := transaction.SetStatementTimeout(checkCtx, timeout); err != nil {
			return err
		}
		catalog, err := transaction.LookupLabel(
			checkCtx, graph.GraphName, label.LabelName,
		)
		if err != nil {
			return err
		}
		expectedKind := age.VertexLabel
		if label.Kind == meta.EdgeLabel {
			expectedKind = age.EdgeLabel
		}
		if catalog.Kind != expectedKind ||
			catalog.NamespaceOID != label.GraphNamespaceOID ||
			catalog.LabelID != label.LabelID ||
			catalog.RelationOID != label.RelationOID ||
			catalog.SequenceOID != label.SequenceOID {
			return fmt.Errorf("label %q catalog identity changed", label.LabelName)
		}
		live.PhysicalRows, err = transaction.VerifyLabelRowsForIdentityCoverage(
			checkCtx,
			catalog,
			count,
			label.Kind == meta.VertexLabel || coverage == identityCoverageFull,
		)
		return err
	})
	checkCancel()
	if err != nil {
		check := classifiedVerificationCheck(id, "label catalog and physical rows were checked", err)
		live.IdentityRows = count
		live.Status = check.Status
		live.Detail = check.Detail
		return check, live
	}
	live.IdentityRows = count
	live.Status = report.CheckPass
	summary := "label catalog, graph IDs, and physical rows match durable identities"
	if label.Kind == meta.EdgeLabel && coverage != identityCoverageFull {
		summary = "label catalog and graph IDs are valid; reverse identity coverage is not asserted"
	}
	return report.Check{
		ID: id, Status: report.CheckPass,
		Summary: summary,
	}, live
}

func countVerificationField(
	label meta.LabelGeneration,
	coverage identityCoverage,
	live liveCountResult,
	counters map[int64]meta.LabelCounter,
	schemaVersion int,
) report.Field {
	name := string(byte(label.Kind)) + "." + label.LabelName
	if live.Status != report.CheckPass {
		return report.Field{Name: name, Value: live.Detail, Status: live.Status}
	}
	if schemaVersion < 17 {
		return report.Field{
			Name: name, Status: report.CheckUnavailable,
			Value: "per-label counters were not recorded by this metadata version",
		}
	}
	stored, ok := counters[label.ID]
	if !ok || stored.Kind != label.Kind {
		return report.Field{
			Name: name, Status: report.CheckUnavailable,
			Value: fmt.Sprintf(
				"identityCoverage=%s; persisted per-label counter is missing",
				reportIdentityCoverage(coverage),
			),
		}
	}
	if stored.Completeness != meta.CounterComplete ||
		stored.Provenance != meta.CounterProvenanceLifecycle {
		return report.Field{
			Name: name, Status: report.CheckUnavailable,
			Value: fmt.Sprintf(
				"counterCompleteness=%s,counterProvenance=%s,identityCoverage=%s; historical per-label values are unavailable",
				stored.Completeness, stored.Provenance,
				reportIdentityCoverage(coverage),
			),
		}
	}
	if stored.AcceptedRows == nil || stored.CommittedRows == nil ||
		stored.RejectedRows == nil {
		return report.Field{
			Name: name, Status: report.CheckUnavailable,
			Value: fmt.Sprintf(
				"identityCoverage=%s; complete persisted per-label counter has unavailable row values",
				reportIdentityCoverage(coverage),
			),
		}
	}
	status := report.CheckPass
	counterComparison := "verified"
	if label.Kind == meta.EdgeLabel && coverage == identityCoverageUnknown {
		status = report.CheckUnavailable
		counterComparison = "unavailable"
	} else if *stored.CommittedRows != live.PhysicalRows {
		status = report.CheckFail
	} else if label.Kind == meta.EdgeLabel && coverage != identityCoverageFull {
		status = report.CheckUnavailable
	}
	bytesValue := "unavailable"
	if stored.CommittedBytes != nil {
		bytesValue = strconv.FormatInt(*stored.CommittedBytes, 10)
	}
	return report.Field{
		Name: name, Status: status,
		Value: fmt.Sprintf(
			"counterCompleteness=%s,counterProvenance=%s,identityCoverage=%s,acceptedRows=%d,committedRows=%d,livePhysicalRows=%d,liveIdentityRows=%d,storedPhysicalComparison=%s,physicalIdentityEquality=%s,committedBytes=%s,rejectedRows=%d",
			stored.Completeness, stored.Provenance,
			reportIdentityCoverage(coverage),
			*stored.AcceptedRows, *stored.CommittedRows,
			live.PhysicalRows, live.IdentityRows,
			counterComparison,
			physicalIdentityEqualityEvidence(label.Kind, coverage),
			bytesValue, *stored.RejectedRows,
		),
	}
}

func reportIdentityCoverage(coverage identityCoverage) string {
	if coverage == identityCoverageUnknown {
		return "legacy-unavailable"
	}
	return string(coverage)
}

func physicalIdentityEqualityEvidence(
	kind meta.LabelKind,
	coverage identityCoverage,
) string {
	if kind == meta.VertexLabel || coverage == identityCoverageFull {
		return "verified"
	}
	return "unavailable"
}

func availabilityEvidence(available bool) string {
	if available {
		return "checked"
	}
	return "unavailable"
}

func integrityVerificationField(
	ctx context.Context,
	adapter *age.Adapter,
	graph meta.GraphGeneration,
	label meta.LabelGeneration,
	coverage identityCoverage,
	limit int,
	timeout time.Duration,
) (report.Field, bool) {
	name := string(byte(label.Kind)) + "." + label.LabelName
	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	var result age.IntegrityResult
	err := adapter.InTransaction(checkCtx, func(transaction *age.Transaction) error {
		if err := transaction.SetStatementTimeout(checkCtx, timeout); err != nil {
			return err
		}
		catalog, err := transaction.LookupLabel(
			checkCtx, graph.GraphName, label.LabelName,
		)
		if err != nil {
			return err
		}
		expectedKind := age.VertexLabel
		if label.Kind == meta.EdgeLabel {
			expectedKind = age.EdgeLabel
		}
		if catalog.Kind != expectedKind ||
			catalog.NamespaceOID != label.GraphNamespaceOID ||
			catalog.LabelID != label.LabelID ||
			catalog.RelationOID != label.RelationOID ||
			catalog.SequenceOID != label.SequenceOID {
			return fmt.Errorf("label %q catalog identity changed", label.LabelName)
		}
		result, err = transaction.VerifyBoundedIntegrityForIdentityCoverage(
			checkCtx, catalog, graph.ID, label.ID, label.Kind, limit,
			label.Kind == meta.VertexLabel || coverage == identityCoverageFull,
		)
		return err
	})
	cancel()
	if err != nil {
		check := classifiedVerificationCheck(
			"integrity."+name, "bounded integrity check could not complete", err,
		)
		return report.Field{Name: name, Value: check.Detail, Status: check.Status}, false
	}
	return integrityResultField(name, coverage, limit, result)
}

func integrityResultField(
	name string,
	coverage identityCoverage,
	limit int,
	result age.IntegrityResult,
) (report.Field, bool) {
	status := report.CheckPass
	if result.MissingPhysicalRows != 0 || result.MissingEndpointRows != 0 ||
		result.ChangedEndpointRows != 0 ||
		(result.PhysicalCoverageChecked && result.OrphanPhysicalRows != 0) {
		status = report.CheckFail
	} else if result.IdentityTruncated || result.PhysicalTruncated {
		status = report.CheckUnknown
	} else if !result.PhysicalCoverageChecked {
		status = report.CheckUnavailable
	}
	return report.Field{
		Name: name, Status: status,
		Value: fmt.Sprintf(
			"limit=%d,identityCoverage=%s,identityRowsChecked=%d,physicalRowsChecked=%d,reversePhysicalCoverage=%s,missingPhysicalRows=%d,orphanPhysicalRows=%d,missingEndpointRows=%d,changedEndpointRows=%d,identityTruncated=%t,physicalTruncated=%t",
			limit, reportIdentityCoverage(coverage),
			result.IdentityRows, result.PhysicalRows,
			availabilityEvidence(result.PhysicalCoverageChecked),
			result.MissingPhysicalRows, result.OrphanPhysicalRows,
			result.MissingEndpointRows, result.ChangedEndpointRows,
			result.IdentityTruncated, result.PhysicalTruncated,
		),
	}, result.IdentityTruncated || result.PhysicalTruncated ||
		!result.PhysicalCoverageChecked
}

func addVerificationMetadataCheck(
	ctx context.Context,
	store *meta.Store,
	schema meta.SchemaInspection,
	job config.LoadJob,
	stored meta.Job,
	timeout time.Duration,
	document *report.Document,
) ([]meta.LabelGeneration, map[int64]identityCoverage, bool) {
	if schema.InstalledVersion < 17 {
		document.Checks = append(document.Checks, report.Check{
			ID: "verification-metadata", Status: report.CheckUnavailable,
			Summary: "2.1 verification metadata is unavailable",
			Detail:  "the job predates metadata schema v17; verify did not migrate it",
		})
		document.IncompleteChecks = append(document.IncompleteChecks, "verification-metadata")
		return nil, nil, false
	}
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	value, err := store.GetJobVerification(readCtx, stored.ID)
	cancel()
	if err != nil {
		document.Checks = append(document.Checks, classifiedVerificationCheck(
			"verification-metadata", "2.1 verification metadata could not be read", err,
		))
		document.IncompleteChecks = append(document.IncompleteChecks, "verification-metadata")
		return nil, nil, false
	}
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		document.Checks = append(document.Checks, report.Check{
			ID: "submitted-configuration", Status: report.CheckFail,
			Summary: "submitted configuration could not be fingerprinted",
		})
		return nil, nil, false
	}
	status := report.CheckPass
	detail := "submitted configuration fingerprint matches the migration snapshot"
	if fingerprint != value.SubmittedConfigFingerprint {
		status = report.CheckFail
		detail = "submitted configuration fingerprint changed"
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "submitted-configuration", Status: status,
		Summary: "submitted configuration fingerprint was checked", Detail: detail,
	})
	snapshot, labels, coverage, err := parseResolvedMappingSummary(
		value.ResolvedMappingSummary,
	)
	if err != nil {
		document.Checks = append(document.Checks, report.Check{
			ID: "resolved-mapping", Status: report.CheckFail,
			Summary: "persisted resolved mapping summary is invalid",
			Detail:  boundedReportValue(err.Error()),
		})
		return nil, nil, false
	}
	canonical, err := json.Marshal(snapshot)
	if err != nil {
		document.Checks = append(document.Checks, report.Check{
			ID: "resolved-mapping", Status: report.CheckFail,
			Summary: "persisted resolved mapping summary could not be canonicalized",
		})
		return nil, nil, false
	}
	digest := sha256.Sum256(canonical)
	resolvedFingerprint := hex.EncodeToString(digest[:])
	mappingStatus := report.CheckPass
	mappingDetail := "resolved mapping fingerprint and source type match the persisted summary"
	if resolvedFingerprint != value.ResolvedMappingFingerprint ||
		snapshot.SourceType != stored.SourceType {
		mappingStatus = report.CheckFail
		mappingDetail = "resolved mapping fingerprint or source type does not match"
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "resolved-mapping", Status: mappingStatus,
		Summary: "persisted resolved mapping snapshot was checked",
		Detail:  mappingDetail,
	})
	return labels, coverage, mappingStatus == report.CheckPass
}

func parseResolvedMappingSummary(
	raw json.RawMessage,
) (
	resolvedMappingSnapshot,
	[]meta.LabelGeneration,
	map[int64]identityCoverage,
	error,
) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var snapshot resolvedMappingSnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return resolvedMappingSnapshot{}, nil, nil,
			fmt.Errorf("decode resolved mapping summary: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return resolvedMappingSnapshot{}, nil, nil, errors.New(
			"resolved mapping summary must contain one JSON object",
		)
	}
	if snapshot.SchemaVersion != legacyResolvedMappingSummaryVersion &&
		snapshot.SchemaVersion != resolvedMappingSummaryVersion {
		return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
			"unsupported resolved mapping summary version %d",
			snapshot.SchemaVersion,
		)
	}
	if strings.TrimSpace(snapshot.SourceType) == "" {
		return resolvedMappingSnapshot{}, nil, nil, errors.New(
			"resolved mapping source type is required",
		)
	}
	if len(snapshot.Labels) == 0 {
		return resolvedMappingSnapshot{}, nil, nil, errors.New(
			"resolved mapping labels are required",
		)
	}
	if len(snapshot.Labels) > maxResolvedMapLabels {
		return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
			"resolved mapping has more than %d labels",
			maxResolvedMapLabels,
		)
	}
	labels := make([]meta.LabelGeneration, 0, len(snapshot.Labels))
	coverage := make(map[int64]identityCoverage, len(snapshot.Labels))
	ids := make(map[int64]struct{}, len(snapshot.Labels))
	names := make(map[string]struct{}, len(snapshot.Labels))
	for _, persisted := range snapshot.Labels {
		var kind meta.LabelKind
		switch persisted.Kind {
		case "v":
			kind = meta.VertexLabel
		case "e":
			kind = meta.EdgeLabel
		default:
			return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
				"resolved label %q has invalid kind %q",
				persisted.Name, persisted.Kind,
			)
		}
		label := meta.LabelGeneration{
			ID:                persisted.ID,
			GraphGenerationID: persisted.GraphGenerationID,
			LabelName:         persisted.Name,
			Kind:              kind,
			GraphNamespaceOID: persisted.GraphNamespaceOID,
			LabelID:           persisted.LabelID,
			RelationOID:       persisted.RelationOID,
			SequenceOID:       persisted.SequenceOID,
			MappingGeneration: persisted.MappingGeneration,
		}
		if err := validateResolvedLabelSnapshot(label); err != nil {
			return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
				"resolved label %q: %w", persisted.Name, err,
			)
		}
		if _, exists := ids[label.ID]; exists {
			return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
				"duplicate resolved label generation %d", label.ID,
			)
		}
		if _, exists := names[label.LabelName]; exists {
			return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
				"duplicate resolved label name %q", label.LabelName,
			)
		}
		resolvedCoverage := persisted.IdentityCoverage
		if snapshot.SchemaVersion == legacyResolvedMappingSummaryVersion {
			if persisted.IdentityCoverage != identityCoverageUnknown {
				return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
					"legacy resolved label %q contains identity coverage",
					persisted.Name,
				)
			}
			if kind == meta.VertexLabel {
				resolvedCoverage = identityCoverageFull
			}
		} else {
			if resolvedCoverage != identityCoverageFull &&
				resolvedCoverage != identityCoverageOptional {
				return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
					"resolved label %q has invalid identity coverage %q",
					persisted.Name,
					persisted.IdentityCoverage,
				)
			}
			if kind == meta.VertexLabel &&
				resolvedCoverage != identityCoverageFull {
				return resolvedMappingSnapshot{}, nil, nil, fmt.Errorf(
					"resolved vertex label %q must have full identity coverage",
					persisted.Name,
				)
			}
		}
		ids[label.ID] = struct{}{}
		names[label.LabelName] = struct{}{}
		coverage[label.ID] = resolvedCoverage
		labels = append(labels, label)
	}
	slices.SortFunc(labels, func(left, right meta.LabelGeneration) int {
		if left.ID < right.ID {
			return -1
		}
		if left.ID > right.ID {
			return 1
		}
		return 0
	})
	return snapshot, labels, coverage, nil
}

func reconcileExpectedLabels(
	expected []meta.LabelGeneration,
	observed []meta.LabelGeneration,
) ([]meta.LabelGeneration, []string) {
	expectedByID := make(map[int64]meta.LabelGeneration, len(expected))
	for _, label := range expected {
		expectedByID[label.ID] = label
	}
	observedByID := make(map[int64]meta.LabelGeneration, len(observed))
	for _, label := range observed {
		if _, expectedLabel := expectedByID[label.ID]; expectedLabel {
			observedByID[label.ID] = label
		}
	}
	matched := make([]meta.LabelGeneration, 0, len(expected))
	mismatches := make([]string, 0)
	for _, wanted := range expected {
		actual, ok := observedByID[wanted.ID]
		if !ok {
			mismatches = append(mismatches, wanted.LabelName+"(missing)")
			continue
		}
		if !sameResolvedLabel(wanted, actual) {
			mismatches = append(mismatches, wanted.LabelName+"(changed)")
			continue
		}
		matched = append(matched, actual)
	}
	slices.SortFunc(matched, func(left, right meta.LabelGeneration) int {
		if left.LabelName != right.LabelName {
			return strings.Compare(left.LabelName, right.LabelName)
		}
		return int(left.Kind) - int(right.Kind)
	})
	slices.Sort(mismatches)
	return matched, mismatches
}

func sameResolvedLabel(left, right meta.LabelGeneration) bool {
	return left.ID == right.ID &&
		left.GraphGenerationID == right.GraphGenerationID &&
		left.LabelName == right.LabelName &&
		left.Kind == right.Kind &&
		left.GraphNamespaceOID == right.GraphNamespaceOID &&
		left.LabelID == right.LabelID &&
		left.RelationOID == right.RelationOID &&
		left.SequenceOID == right.SequenceOID &&
		left.MappingGeneration == right.MappingGeneration
}

func addVerificationLabelLimit(
	document *report.Document,
	options VerifyOptions,
	expected int,
) {
	document.Checks = append(document.Checks, report.Check{
		ID: "resolved-label-limit", Status: report.CheckUnknown,
		Summary: "persisted resolved label set exceeds the verification limit",
		Detail: fmt.Sprintf(
			"expectedLabels=%d,limit=%d; no label counts or integrity checks were run",
			expected, maxVerifyLabels,
		),
	})
	document.IncompleteChecks = append(document.IncompleteChecks, "label-generations")
	document.Warnings = append(document.Warnings, report.Finding{
		Code: "LABEL_LIMIT_EXCEEDED",
		Message: fmt.Sprintf(
			"the persisted resolved set has %d labels; verification is limited to %d",
			expected, maxVerifyLabels,
		),
	})
	if options.Counts {
		document.Sections = append(document.Sections, report.Section{
			Title: "Per-label counts",
			Fields: []report.Field{{
				Name: "expected-labels", Status: report.CheckUnknown,
				Value: "not checked because the persisted resolved label set exceeds the limit",
			}},
		})
	}
	if options.Integrity {
		document.Sections = append(document.Sections, report.Section{
			Title: "Bounded integrity",
			Fields: []report.Field{{
				Name: "expected-labels", Status: report.CheckUnknown,
				Value: "not checked because the persisted resolved label set exceeds the limit",
			}},
		})
	}
}

func unclassifiedRejectField(
	ctx context.Context,
	store *meta.Store,
	schemaVersion int,
	jobID string,
	timeout time.Duration,
) report.Field {
	if schemaVersion < 17 {
		return report.Field{
			Name: "unclassified.rejects", Status: report.CheckUnavailable,
			Value: "unclassified reject counters were not recorded by this metadata version",
		}
	}
	readCtx, cancel := context.WithTimeout(ctx, timeout)
	count, err := store.GetUnclassifiedRejects(readCtx, jobID)
	cancel()
	if err != nil {
		check := classifiedVerificationCheck(
			"unclassified.rejects", "unclassified reject counter could not be read", err,
		)
		return report.Field{
			Name: "unclassified.rejects", Status: check.Status, Value: check.Detail,
		}
	}
	return report.Field{
		Name: "unclassified.rejects", Status: report.CheckPass,
		Value: strconv.FormatInt(count, 10),
	}
}

func classifiedVerificationCheck(id, summary string, err error) report.Check {
	status := report.CheckFail
	detail := boundedReportValue(err.Error())
	var pgErr *pgconn.PgError
	switch {
	case errors.Is(err, meta.ErrNotFound):
		status = report.CheckUnavailable
		detail = "required persisted verification metadata is unavailable"
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		status = report.CheckUnknown
		detail = "verification did not complete before its operation deadline"
	case errors.As(err, &pgErr) && pgErr.Code == "57014":
		status = report.CheckUnknown
		detail = "verification was canceled by the statement timeout"
	case errors.As(err, &pgErr) && pgErr.Code == "42501":
		status = report.CheckUnknown
		detail = "permission denied while running verification"
	}
	return report.Check{ID: id, Status: status, Summary: summary, Detail: detail}
}

func validatedVerificationReport(document report.Document) (report.Document, error) {
	if document.Outcome == "" {
		document.Outcome = report.OutcomePass
		if hasStatus(document, report.CheckFail) {
			document.Outcome = report.OutcomeFail
		} else if hasStatus(document, report.CheckUnknown) ||
			hasStatus(document, report.CheckUnavailable) ||
			len(document.IncompleteChecks) > 0 {
			document.Outcome = report.OutcomeIncomplete
		}
	}
	if _, err := report.Render(document, report.FormatJSON); err != nil {
		return report.Document{}, err
	}
	return document, nil
}
