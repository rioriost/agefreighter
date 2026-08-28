package app

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

const (
	MaxOptimizeLabels          = 64
	MaxOptimizeIndexesPerTable = 64
	MaxOptimizeBatchAttempts   = 1000
	MaxOptimizeIndexFindings   = 128
	MaxOptimizeRecommendations = 128
)

const propertyEvidenceUnavailable = "Apache AGE 1.6 cannot pre-bound agtype serialization before detoast; live property parsing, cardinality inspection, and property-index recommendations are disabled"

var errOptimizerSavepointRecovery = errors.New(
	"optimizer inspection savepoint could not be recovered",
)
var errOptimizerSavepointControl = errors.New(
	"optimizer inspection savepoint control failed",
)
var errOptimizerUnknownSchema = errors.New(
	"optimizer metadata schema visibility is unknown",
)

type OptimizeOptions struct {
	Analyze     bool
	GeneratedAt time.Time
}

type optimizationSnapshot struct {
	Probe                    age.DegradedProbe
	Schema                   meta.SchemaInspection
	Graph                    meta.GraphGeneration
	GraphAvailable           bool
	GraphStatus              report.CheckStatus
	GraphDetail              string
	Labels                   []meta.LabelGeneration
	LabelsTruncated          bool
	Relations                []relationEvidence
	MetadataRelations        []relationEvidence
	MetadataRelationsMissing []string
	RequiredMetadataInvalid  []string
	MetadataIndexStatus      report.CheckStatus
	Job                      meta.Job
	JobAvailable             bool
	MigrationStatus          report.CheckStatus
	MigrationDetail          string
	LatestBatch              meta.BatchAttempt
	LatestBatchAvailable     bool
	BatchAttemptsObserved    int
	BatchAttemptsTruncated   bool
	BatchAttemptsStatus      report.CheckStatus
	Telemetry                meta.ConnectorTelemetry
	TelemetryAvailable       bool
	Counters                 []meta.LabelCounter
	CountersAvailable        bool
	DatabaseBytes            int64
	DatabaseSizeStatus       report.CheckStatus
	WALBytes                 int64
	WALReset                 *time.Time
	WALStatus                report.CheckStatus
	StatsReset               *time.Time
	StatsResetStatus         report.CheckStatus
	GINStatus                report.CheckStatus
	GINSupported             bool
	AnalyzeResults           []analyzeResult
}

type relationEvidence struct {
	Schema              string
	Name                string
	OID                 uint32
	Kind                meta.LabelKind
	GraphOID            uint32
	GraphNamespaceOID   uint32
	GraphGenerationID   int64
	LabelGenerationID   int64
	LabelID             uint16
	SequenceOID         uint32
	MappingGeneration   uint64
	EstimatedRows       float64
	TotalBytes          int64
	IndexBytes          int64
	LiveRows            int64
	DeadRows            int64
	SequentialScans     int64
	IndexScans          int64
	LastAnalyze         *time.Time
	LastAutoAnalyze     *time.Time
	LastVacuum          *time.Time
	LastAutoVacuum      *time.Time
	GenerationUpdatedAt *time.Time
	Status              report.CheckStatus
	Detail              string
	Indexes             []indexEvidence
	IndexesTruncated    bool
	RequiredIndexStatus report.CheckStatus
	RequiredIndexAbsent []string
}

type indexEvidence struct {
	Name         string
	Signature    string
	AccessMethod string
	Predicate    string
	Valid        bool
	Ready        bool
	Unique       bool
	Primary      bool
	KeyNames     []string
	KeyOptions   []int16
	Scans        int64
}

type analyzeResult struct {
	Scope  string
	Status string
	Detail string
}

func OptimizationReport(
	ctx context.Context,
	path string,
	options OptimizeOptions,
) (report.Document, error) {
	job, err := config.Load(path)
	if err != nil {
		return report.Document{}, fmt.Errorf("load target configuration: %w", err)
	}
	timeout := time.Duration(job.Runtime.OperationTimeout)
	evidenceCtx, cancelEvidence := context.WithTimeout(ctx, timeout)

	probe, err := probeTarget(evidenceCtx, job)
	if err != nil {
		cancelEvidence()
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return report.Document{}, canceledErr
		}
		return report.Document{}, errors.New("probe optimizer target")
	}
	if err := evidenceCtx.Err(); err != nil {
		cancelEvidence()
		return report.Document{}, err
	}
	if probe.PostgreSQLStatus != age.ProbePass ||
		probe.AGEPresenceStatus != age.ProbePass ||
		probe.AGEVersionStatus != age.ProbePass ||
		probe.AGELoadabilityStatus != age.ProbePass {
		cancelEvidence()
		return report.Document{}, errors.New(
			"optimizer requires a compatible, loadable PostgreSQL 17 and Apache AGE 1.6 target",
		)
	}
	dsn, err := resolveSecret(job.Target.Connection)
	if err != nil {
		cancelEvidence()
		return report.Document{}, fmt.Errorf("resolve target connection: %w", err)
	}
	pool, err := openDoctorPool(evidenceCtx, dsn)
	if err != nil {
		cancelEvidence()
		if canceledErr := canceled(ctx, err); canceledErr != nil {
			return report.Document{}, canceledErr
		}
		return report.Document{}, errors.New("open read-only optimizer target")
	}
	snapshot, err := collectOptimizationEvidence(evidenceCtx, pool, job, probe)
	pool.Close()
	cancelEvidence()
	if err != nil {
		return report.Document{}, err
	}
	if options.Analyze {
		if err := validateAnalyzePreconditions(snapshot); err != nil {
			return report.Document{}, err
		}
		openCtx, cancelOpen := context.WithTimeout(ctx, timeout)
		analyzePool, openErr := openAnalyzePool(openCtx, dsn, timeout)
		cancelOpen()
		if openErr != nil {
			return report.Document{}, openErr
		}
		snapshot.AnalyzeResults, err = applyBoundedAnalyze(
			ctx,
			analyzePool,
			snapshot,
			timeout,
		)
		analyzePool.Close()
		if err != nil {
			return report.Document{}, err
		}
	}
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	return buildOptimizationReport(snapshot, options.Analyze, at)
}

func collectOptimizationEvidence(
	ctx context.Context,
	pool *pgxpool.Pool,
	job config.LoadJob,
	probe age.DegradedProbe,
) (optimizationSnapshot, error) {
	snapshot := optimizationSnapshot{
		Probe:               probe,
		DatabaseSizeStatus:  report.CheckUnknown,
		WALStatus:           report.CheckUnknown,
		StatsResetStatus:    report.CheckUnknown,
		GINStatus:           report.CheckUnknown,
		GraphStatus:         report.CheckUnavailable,
		MetadataIndexStatus: report.CheckPass,
		MigrationStatus:     report.CheckUnavailable,
		BatchAttemptsStatus: report.CheckUnavailable,
	}
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{
		IsoLevel:   pgx.RepeatableRead,
		AccessMode: pgx.ReadOnly,
	})
	if err != nil {
		return snapshot, errors.New("begin read-only optimizer inspection")
	}
	defer rollbackOptimizerTx(tx)
	if _, err := tx.Exec(ctx, `
		SELECT pg_catalog.set_config('statement_timeout', $1, true)`,
		postgresDuration(optimizerRemaining(ctx)),
	); err != nil {
		return snapshot, classifyOptimizerFatal(
			ctx,
			"set optimizer inspection statement timeout",
			err,
		)
	}
	err = runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		probeStore, storeErr := meta.New(probeTx)
		if storeErr != nil {
			return storeErr
		}
		snapshot.Schema, storeErr = probeStore.InspectSchema(ctx)
		if storeErr == nil && snapshot.Schema.State == meta.SchemaUnknown {
			return errOptimizerUnknownSchema
		}
		return storeErr
	})
	if err != nil && !errors.Is(err, errOptimizerUnknownSchema) {
		return snapshot, classifyOptimizerFatal(
			ctx,
			"inspect optimizer metadata schema",
			err,
		)
	}
	if err := snapshot.Schema.RequireReadCompatible(); err != nil {
		if snapshot.Schema.State == meta.SchemaUnknown {
			snapshot.GraphStatus = report.CheckUnknown
			snapshot.GraphDetail = "active graph ownership is unknown because metadata visibility is unavailable"
			snapshot.MetadataIndexStatus = report.CheckUnknown
			snapshot.MigrationStatus = report.CheckUnknown
			snapshot.MigrationDetail = "migration evidence is unknown because metadata visibility is unavailable"
			if commitErr := tx.Commit(ctx); commitErr != nil {
				return snapshot, classifyOptimizerFatal(
					ctx,
					"finish metadata-version inspection",
					commitErr,
				)
			}
			return snapshot, nil
		}
		return snapshot, err
	}

	var graphs meta.GraphGenerationPage
	err = runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		probeStore, storeErr := meta.New(probeTx)
		if storeErr != nil {
			return storeErr
		}
		var graphErr error
		graphs, graphErr = probeStore.ListCurrentGraphGenerations(
			ctx,
			job.Target.Graph,
			3,
		)
		return graphErr
	})
	if err != nil {
		if fatalErr := optimizerProbeFatal(ctx, err); fatalErr != nil {
			return snapshot, fatalErr
		}
		if optimizerEvidenceUnknown(err) {
			snapshot.GraphStatus = report.CheckUnknown
			snapshot.GraphDetail = safeDatabaseDetail(
				err,
				"active graph generation visibility is unknown",
			)
		} else {
			return snapshot, classifyOptimizerFatal(ctx, "read active graph generation", err)
		}
	}
	for _, graph := range graphs.Generations {
		if graph.State == meta.GenerationActive && graph.GraphName == job.Target.Graph {
			if snapshot.GraphAvailable {
				return snapshot, errors.New("multiple active graph generations were found")
			}
			snapshot.Graph = graph
			snapshot.GraphAvailable = true
			snapshot.GraphStatus = report.CheckPass
			snapshot.GraphDetail = "one active agefreighter-owned graph generation was selected"
		}
	}
	if err == nil && !graphs.Complete {
		snapshot.GraphStatus = report.CheckUnknown
		snapshot.GraphDetail = "active graph generation visibility exceeded the bounded catalog limit"
	}
	if snapshot.GraphAvailable {
		var labels []meta.LabelGeneration
		labelErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			probeStore, storeErr := meta.New(probeTx)
			if storeErr != nil {
				return storeErr
			}
			var readErr error
			labels, readErr = probeStore.ListLabelGenerations(
				ctx,
				snapshot.Graph.ID,
				MaxOptimizeLabels+1,
			)
			return readErr
		})
		if labelErr != nil {
			if fatalErr := optimizerProbeFatal(ctx, labelErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			if optimizerEvidenceUnknown(labelErr) {
				snapshot.GraphStatus = report.CheckUnknown
				snapshot.GraphDetail = safeDatabaseDetail(
					labelErr,
					"active label generation visibility is unknown",
				)
			} else {
				return snapshot, classifyOptimizerFatal(ctx, "read active label generations", labelErr)
			}
			labels = nil
		}
		if len(labels) > MaxOptimizeLabels {
			snapshot.LabelsTruncated = true
			labels = labels[:MaxOptimizeLabels]
		}
		snapshot.Labels = labels
	}

	if snapshot.GraphAvailable {
		var stored meta.Job
		jobErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			probeStore, storeErr := meta.New(probeTx)
			if storeErr != nil {
				return storeErr
			}
			var readErr error
			stored, readErr = probeStore.GetJob(ctx, snapshot.Graph.JobID)
			return readErr
		})
		if jobErr == nil {
			snapshot.Job = stored
			snapshot.JobAvailable = true
			snapshot.MigrationStatus = report.CheckPass
		} else if !errors.Is(jobErr, meta.ErrNotFound) {
			if fatalErr := optimizerProbeFatal(ctx, jobErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			if optimizerEvidenceUnknown(jobErr) {
				snapshot.MigrationStatus = report.CheckUnknown
				snapshot.MigrationDetail = safeDatabaseDetail(
					jobErr,
					"migration counter visibility is unknown",
				)
			} else {
				return snapshot, classifyOptimizerFatal(ctx, "read migration counters", jobErr)
			}
		}
	}
	if snapshot.JobAvailable {
		var boundedAttempts int
		attemptErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			return probeTx.QueryRow(ctx, `
				SELECT COUNT(*)::integer
				FROM (
					SELECT 1
					FROM agefreighter_meta.load_batch
					WHERE job_id = $1::uuid
					ORDER BY batch_id, attempt
					LIMIT $2 + 1
				) bounded_attempts`,
				snapshot.Job.ID,
				MaxOptimizeBatchAttempts,
			).Scan(&boundedAttempts)
		})
		if attemptErr == nil {
			snapshot.BatchAttemptsTruncated =
				boundedAttempts > MaxOptimizeBatchAttempts
			snapshot.BatchAttemptsObserved = min(
				boundedAttempts,
				MaxOptimizeBatchAttempts,
			)
			snapshot.BatchAttemptsStatus = report.CheckPass
		} else if optimizerEvidenceUnknown(attemptErr) {
			if fatalErr := optimizerProbeFatal(ctx, attemptErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			snapshot.MigrationStatus = report.CheckUnknown
			snapshot.MigrationDetail = safeDatabaseDetail(
				attemptErr,
				"batch-attempt visibility is unknown",
			)
		} else {
			return snapshot, classifyOptimizerFatal(ctx, "read bounded batch attempts", attemptErr)
		}
		var latest meta.BatchAttempt
		batchErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			probeStore, storeErr := meta.New(probeTx)
			if storeErr != nil {
				return storeErr
			}
			var readErr error
			latest, readErr = probeStore.LatestBatch(ctx, snapshot.Job.ID)
			return readErr
		})
		if batchErr == nil {
			snapshot.LatestBatch = latest
			snapshot.LatestBatchAvailable = true
		} else if !errors.Is(batchErr, meta.ErrNotFound) {
			if fatalErr := optimizerProbeFatal(ctx, batchErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			if optimizerEvidenceUnknown(batchErr) {
				snapshot.MigrationStatus = report.CheckUnknown
				snapshot.MigrationDetail = safeDatabaseDetail(
					batchErr,
					"latest batch telemetry visibility is unknown",
				)
			} else {
				return snapshot, classifyOptimizerFatal(ctx, "read latest batch telemetry", batchErr)
			}
		}
		if snapshot.Schema.InstalledVersion >= 15 {
			var telemetry meta.ConnectorTelemetry
			telemetryErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
				probeStore, storeErr := meta.New(probeTx)
				if storeErr != nil {
					return storeErr
				}
				var readErr error
				telemetry, readErr = probeStore.GetConnectorTelemetry(
					ctx,
					snapshot.Job.ID,
				)
				return readErr
			})
			if telemetryErr == nil {
				snapshot.Telemetry = telemetry
				snapshot.TelemetryAvailable = true
			} else if !errors.Is(telemetryErr, meta.ErrNotFound) {
				if fatalErr := optimizerProbeFatal(ctx, telemetryErr); fatalErr != nil {
					return snapshot, fatalErr
				}
				if optimizerEvidenceUnknown(telemetryErr) {
					snapshot.MigrationStatus = report.CheckUnknown
					snapshot.MigrationDetail = safeDatabaseDetail(
						telemetryErr,
						"connector telemetry visibility is unknown",
					)
				} else {
					return snapshot, classifyOptimizerFatal(ctx, "read connector telemetry", telemetryErr)
				}
			}
		}
		if snapshot.Schema.InstalledVersion >= 17 {
			var counters []meta.LabelCounter
			counterErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
				probeStore, storeErr := meta.New(probeTx)
				if storeErr != nil {
					return storeErr
				}
				var readErr error
				counters, readErr = probeStore.ListLabelCounters(
					ctx,
					snapshot.Job.ID,
					MaxOptimizeLabels+1,
				)
				return readErr
			})
			if counterErr != nil {
				if fatalErr := optimizerProbeFatal(ctx, counterErr); fatalErr != nil {
					return snapshot, fatalErr
				}
				if optimizerEvidenceUnknown(counterErr) {
					snapshot.MigrationStatus = report.CheckUnknown
					snapshot.MigrationDetail = safeDatabaseDetail(
						counterErr,
						"per-label counter visibility is unknown",
					)
					counters = nil
				} else {
					return snapshot, classifyOptimizerFatal(ctx, "read label counters", counterErr)
				}
			}
			if len(counters) > MaxOptimizeLabels {
				counters = counters[:MaxOptimizeLabels]
				snapshot.LabelsTruncated = true
			}
			if counterErr == nil {
				snapshot.Counters = counters
				snapshot.CountersAvailable = true
			}
		}
	}

	databaseSizeErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		return probeTx.QueryRow(
			ctx,
			`SELECT pg_catalog.pg_database_size(current_database())`,
		).Scan(&snapshot.DatabaseBytes)
	})
	if databaseSizeErr == nil {
		snapshot.DatabaseSizeStatus = report.CheckPass
	} else if fatalErr := optimizerProbeFatal(ctx, databaseSizeErr); fatalErr != nil {
		return snapshot, fatalErr
	}
	walErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		return probeTx.QueryRow(ctx, `
			SELECT wal_bytes::bigint, stats_reset
			FROM pg_catalog.pg_stat_wal`,
		).Scan(&snapshot.WALBytes, &snapshot.WALReset)
	})
	if walErr == nil {
		snapshot.WALStatus = report.CheckPass
	} else if fatalErr := optimizerProbeFatal(ctx, walErr); fatalErr != nil {
		return snapshot, fatalErr
	}
	statsResetErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		return probeTx.QueryRow(ctx, `
			SELECT stats_reset
			FROM pg_catalog.pg_stat_database
			WHERE datname = current_database()`,
		).Scan(&snapshot.StatsReset)
	})
	if statsResetErr == nil {
		snapshot.StatsResetStatus = report.CheckPass
	} else if fatalErr := optimizerProbeFatal(ctx, statsResetErr); fatalErr != nil {
		return snapshot, fatalErr
	}
	var supportedGIN bool
	ginErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		return probeTx.QueryRow(ctx, `
			SELECT EXISTS (
				SELECT 1
				FROM pg_catalog.pg_opclass operator_class
				JOIN pg_catalog.pg_am access_method
				  ON access_method.oid = operator_class.opcmethod
				JOIN pg_catalog.pg_namespace namespace
				  ON namespace.oid = operator_class.opcnamespace
				JOIN pg_catalog.pg_type value_type
				  ON value_type.oid = operator_class.opcintype
				WHERE access_method.amname = 'gin'
				  AND namespace.nspname = 'ag_catalog'
				  AND value_type.typname = 'agtype'
				  AND operator_class.opcname IN ('agtype_ops')
			)`,
		).Scan(&supportedGIN)
	})
	if ginErr == nil {
		snapshot.GINStatus = report.CheckPass
		snapshot.GINSupported = supportedGIN
	} else if fatalErr := optimizerProbeFatal(ctx, ginErr); fatalErr != nil {
		return snapshot, fatalErr
	}

	for _, label := range snapshot.Labels {
		var relation relationEvidence
		relationErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			var inspectErr error
			relation, inspectErr = inspectOptimizationRelation(
				ctx,
				probeTx,
				job.Target.Graph,
				label.LabelName,
				label.RelationOID,
				label.Kind,
				true,
			)
			return inspectErr
		})
		if relationErr != nil {
			if fatalErr := optimizerProbeFatal(ctx, relationErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			relation = relationEvidence{
				Schema: job.Target.Graph,
				Name:   label.LabelName,
				OID:    label.RelationOID,
				Kind:   label.Kind,
				Status: report.CheckUnknown,
				Detail: safeDatabaseDetail(
					relationErr,
					"catalog or statistics visibility is unavailable",
				),
			}
		}
		relation.GenerationUpdatedAt = &label.UpdatedAt
		relation.GraphOID = snapshot.Graph.GraphOID
		relation.GraphNamespaceOID = label.GraphNamespaceOID
		relation.GraphGenerationID = label.GraphGenerationID
		relation.LabelGenerationID = label.ID
		relation.LabelID = label.LabelID
		relation.SequenceOID = label.SequenceOID
		relation.MappingGeneration = label.MappingGeneration
		snapshot.Relations = append(snapshot.Relations, relation)
	}

	metadataNames := optimizerMetadataAllowlist(snapshot.Schema.InstalledVersion)
	metadataOIDs, missing, err := inspectMetadataRelationOIDs(ctx, tx, metadataNames)
	if err != nil {
		return snapshot, classifyOptimizerFatal(ctx, "inspect metadata relations", err)
	}
	snapshot.MetadataRelationsMissing = missing
	for _, name := range metadataNames {
		oid, exists := metadataOIDs[name]
		if !exists {
			continue
		}
		var relation relationEvidence
		relationErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			var inspectErr error
			relation, inspectErr = inspectOptimizationRelation(
				ctx, probeTx, "agefreighter_meta", name, oid, 0, false,
			)
			return inspectErr
		})
		if relationErr != nil {
			if fatalErr := optimizerProbeFatal(ctx, relationErr); fatalErr != nil {
				return snapshot, fatalErr
			}
			relation = relationEvidence{
				Schema: "agefreighter_meta", Name: name, OID: oid,
				Status: report.CheckUnknown,
				Detail: safeDatabaseDetail(
					relationErr,
					"catalog or statistics visibility is unavailable",
				),
			}
		}
		snapshot.MetadataRelations = append(snapshot.MetadataRelations, relation)
	}
	definitions := requiredMetadataIndexesForVersion(snapshot.Schema.InstalledVersion)
	var indexes []metadataIndexInspection
	err = runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
		var inspectErr error
		indexes, inspectErr = inspectMetadataIndexes(
			ctx,
			probeTx,
			optimizerRemaining(ctx),
			definitions,
		)
		return inspectErr
	})
	if err != nil {
		if fatalErr := optimizerProbeFatal(ctx, err); fatalErr != nil {
			return snapshot, fatalErr
		}
		snapshot.RequiredMetadataInvalid = make([]string, len(definitions))
		snapshot.MetadataIndexStatus = report.CheckUnknown
		for index, definition := range definitions {
			snapshot.RequiredMetadataInvalid[index] = definition.Name
		}
	} else {
		snapshot.RequiredMetadataInvalid = validateMetadataIndexDefinitions(
			indexes,
			definitions,
		)
	}

	if err := tx.Commit(ctx); err != nil {
		return snapshot, classifyOptimizerFatal(ctx, "finish read-only optimizer inspection", err)
	}
	return snapshot, nil
}

func inspectOptimizationRelation(
	ctx context.Context,
	tx pgx.Tx,
	expectedSchema string,
	expectedName string,
	expectedOID uint32,
	kind meta.LabelKind,
	inspectAGEIndexes bool,
) (relationEvidence, error) {
	value := relationEvidence{
		Schema:              expectedSchema,
		Name:                expectedName,
		OID:                 expectedOID,
		Kind:                kind,
		Status:              report.CheckPass,
		RequiredIndexStatus: report.CheckUnavailable,
	}
	var observedOID uint32
	err := tx.QueryRow(ctx, `
		SELECT
			namespace.nspname,
			relation.relname,
			relation.oid,
			relation.reltuples::double precision,
			pg_catalog.pg_total_relation_size(relation.oid),
			pg_catalog.pg_indexes_size(relation.oid),
			COALESCE(statistics.n_live_tup, 0),
			COALESCE(statistics.n_dead_tup, 0),
			COALESCE(statistics.seq_scan, 0),
			COALESCE(statistics.idx_scan, 0),
			statistics.last_analyze,
			statistics.last_autoanalyze,
			statistics.last_vacuum,
			statistics.last_autovacuum
		FROM pg_catalog.pg_class relation
		JOIN pg_catalog.pg_namespace namespace
		  ON namespace.oid = relation.relnamespace
		LEFT JOIN pg_catalog.pg_stat_all_tables statistics
		  ON statistics.relid = relation.oid
		WHERE relation.oid = $1
		  AND relation.relkind IN ('r', 'p')`,
		expectedOID,
	).Scan(
		&value.Schema,
		&value.Name,
		&observedOID,
		&value.EstimatedRows,
		&value.TotalBytes,
		&value.IndexBytes,
		&value.LiveRows,
		&value.DeadRows,
		&value.SequentialScans,
		&value.IndexScans,
		&value.LastAnalyze,
		&value.LastAutoAnalyze,
		&value.LastVacuum,
		&value.LastAutoVacuum,
	)
	if err != nil {
		return value, err
	}
	if observedOID != expectedOID ||
		value.Schema != expectedSchema ||
		value.Name != expectedName {
		return value, errors.New("relation catalog identity changed during inspection")
	}
	indexes, truncated, err := inspectRelationIndexes(ctx, tx, expectedOID)
	if err != nil {
		return value, err
	}
	value.Indexes = indexes
	value.IndexesTruncated = truncated
	if inspectAGEIndexes {
		requiredErr := runOptimizerProbe(ctx, tx, func(probeTx pgx.Tx) error {
			var inspectErr error
			value.RequiredIndexAbsent, inspectErr = inspectRequiredAGEIndexes(
				ctx,
				probeTx,
				expectedOID,
				kind,
			)
			return inspectErr
		})
		if requiredErr != nil {
			if fatalErr := optimizerProbeFatal(ctx, requiredErr); fatalErr != nil {
				return value, fatalErr
			}
			value.Status = report.CheckUnknown
			value.RequiredIndexStatus = report.CheckUnknown
			value.Detail = safeDatabaseDetail(
				requiredErr,
				"required AGE index visibility is unavailable",
			)
			return value, nil
		}
		value.RequiredIndexStatus = report.CheckPass
	}
	return value, nil
}

const requiredAGEIndexesSQL = `
	SELECT
		required.column_name,
		EXISTS (
			SELECT 1
			FROM pg_catalog.pg_index index_metadata
			JOIN pg_catalog.pg_class index_relation
			  ON index_relation.oid = index_metadata.indexrelid
			JOIN pg_catalog.pg_am access_method
			  ON access_method.oid = index_relation.relam
			JOIN pg_catalog.pg_attribute attribute
			  ON attribute.attrelid = index_metadata.indrelid
			 AND attribute.attname = required.column_name
			 AND NOT attribute.attisdropped
			WHERE index_metadata.indrelid = $1
			  AND index_metadata.indisvalid
			  AND index_metadata.indisready
			  AND index_metadata.indislive
			  AND NOT index_metadata.indisexclusion
			  AND index_metadata.indimmediate
			  AND NOT index_metadata.indnullsnotdistinct
			  AND access_method.amname = 'btree'
			  AND index_metadata.indpred IS NULL
			  AND index_metadata.indexprs IS NULL
			  AND index_metadata.indnatts = 1
			  AND index_metadata.indnkeyatts = 1
			  AND ARRAY(
					SELECT key_value
					FROM unnest(index_metadata.indkey::smallint[]) key_value
			  ) = ARRAY[attribute.attnum::smallint]
			  AND ARRAY(
					SELECT option_value
					FROM unnest(index_metadata.indoption::smallint[]) option_value
			  ) = ARRAY[0::smallint]
			  AND index_metadata.indisunique =
					(required.column_name = 'id')
			  AND index_metadata.indisprimary =
					(required.column_name = 'id')
		) AS exact_definition_exists
	FROM unnest($2::text[]) required(column_name)
	ORDER BY required.column_name`

func inspectRequiredAGEIndexes(
	ctx context.Context,
	tx pgx.Tx,
	relationOID uint32,
	kind meta.LabelKind,
) ([]string, error) {
	required := []string{"id"}
	if kind == meta.EdgeLabel {
		required = append(required, "start_id", "end_id")
	}
	rows, err := tx.Query(
		ctx,
		requiredAGEIndexesSQL,
		relationOID,
		required,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	found := make(map[string]bool, len(required))
	for rows.Next() {
		var column string
		var exists bool
		if err := rows.Scan(&column, &exists); err != nil {
			return nil, err
		}
		found[column] = exists
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	missing := make([]string, 0, len(required))
	for _, column := range required {
		if !found[column] {
			missing = append(missing, column)
		}
	}
	return missing, nil
}

func inspectRelationIndexes(
	ctx context.Context,
	tx pgx.Tx,
	relationOID uint32,
) ([]indexEvidence, bool, error) {
	rows, err := tx.Query(ctx, `
		SELECT
			index_relation.relname,
			pg_catalog.md5(pg_catalog.concat_ws(
				'|',
				index_metadata.indnatts::text,
				index_metadata.indnkeyatts::text,
				index_metadata.indisunique::text,
				index_metadata.indisprimary::text,
				index_metadata.indisexclusion::text,
				index_metadata.indimmediate::text,
				index_metadata.indnullsnotdistinct::text,
				index_metadata.indkey::text,
				index_metadata.indclass::text,
				index_metadata.indcollation::text,
				index_metadata.indoption::text,
				COALESCE(pg_catalog.pg_get_expr(
					index_metadata.indexprs,
					index_metadata.indrelid
				), ''),
				COALESCE(pg_catalog.pg_get_expr(
					index_metadata.indpred,
					index_metadata.indrelid
				), ''),
				index_relation.reltablespace::text,
				COALESCE(
					pg_catalog.array_to_string(index_relation.reloptions, ','),
					''
				),
				access_method.amname
			)),
			index_metadata.indisvalid,
			index_metadata.indisready,
			index_metadata.indisunique,
			index_metadata.indisprimary,
			access_method.amname,
			COALESCE(pg_catalog.pg_get_expr(
				index_metadata.indpred,
				index_metadata.indrelid,
				true
			), ''),
			ARRAY(
				SELECT COALESCE(attribute.attname, '')
				FROM unnest(index_metadata.indkey::smallint[])
				     WITH ORDINALITY AS key(attnum, position)
				LEFT JOIN pg_catalog.pg_attribute attribute
				  ON attribute.attrelid = index_metadata.indrelid
				 AND attribute.attnum = key.attnum
				WHERE key.position <= index_metadata.indnkeyatts
				ORDER BY key.position
			),
			ARRAY(
				SELECT option.value
				FROM unnest(index_metadata.indoption::smallint[])
				     WITH ORDINALITY AS option(value, position)
				WHERE option.position <= index_metadata.indnkeyatts
				ORDER BY option.position
			),
			COALESCE(index_statistics.idx_scan, 0)
		FROM pg_catalog.pg_index index_metadata
		JOIN pg_catalog.pg_class index_relation
		  ON index_relation.oid = index_metadata.indexrelid
		JOIN pg_catalog.pg_class table_relation
		  ON table_relation.oid = index_metadata.indrelid
		JOIN pg_catalog.pg_am access_method
		  ON access_method.oid = index_relation.relam
		LEFT JOIN pg_catalog.pg_stat_all_indexes index_statistics
		  ON index_statistics.indexrelid = index_metadata.indexrelid
		WHERE index_metadata.indrelid = $1
		ORDER BY index_relation.relname
		LIMIT $2`,
		relationOID,
		MaxOptimizeIndexesPerTable+1,
	)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()
	values := make([]indexEvidence, 0, MaxOptimizeIndexesPerTable)
	truncated := false
	for rows.Next() {
		var value indexEvidence
		if err := rows.Scan(
			&value.Name,
			&value.Signature,
			&value.Valid,
			&value.Ready,
			&value.Unique,
			&value.Primary,
			&value.AccessMethod,
			&value.Predicate,
			&value.KeyNames,
			&value.KeyOptions,
			&value.Scans,
		); err != nil {
			return nil, false, err
		}
		if len(values) == MaxOptimizeIndexesPerTable {
			truncated = true
			continue
		}
		values = append(values, value)
	}
	return values, truncated, rows.Err()
}

func requiredAGEIndexColumns(
	kind meta.LabelKind,
	indexes []indexEvidence,
) []string {
	required := []string{"id"}
	if kind == meta.EdgeLabel {
		required = append(required, "start_id", "end_id")
	}
	missing := make([]string, 0, len(required))
	for _, column := range required {
		found := false
		for _, index := range indexes {
			requireUniquePrimary := column == "id"
			if index.Valid &&
				index.Ready &&
				index.AccessMethod == "btree" &&
				index.Predicate == "" &&
				slices.Equal(index.KeyNames, []string{column}) &&
				slices.Equal(index.KeyOptions, []int16{0}) &&
				index.Unique == requireUniquePrimary &&
				index.Primary == requireUniquePrimary {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, column)
		}
	}
	return missing
}

func inspectMetadataRelationOIDs(
	ctx context.Context,
	tx pgx.Tx,
	expected []string,
) (map[string]uint32, []string, error) {
	rows, err := tx.Query(ctx, `
		SELECT relation.relname, relation.oid
		FROM pg_catalog.pg_class relation
		JOIN pg_catalog.pg_namespace namespace
		  ON namespace.oid = relation.relnamespace
		WHERE namespace.nspname = 'agefreighter_meta'
		  AND relation.relkind IN ('r', 'p')
		  AND relation.relname = ANY($1::text[])
		ORDER BY relation.relname`,
		expected,
	)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	found := make(map[string]uint32, len(expected))
	for rows.Next() {
		var name string
		var oid uint32
		if err := rows.Scan(&name, &oid); err != nil {
			return nil, nil, err
		}
		found[name] = oid
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	missing := make([]string, 0)
	for _, name := range expected {
		if _, exists := found[name]; !exists {
			missing = append(missing, name)
		}
	}
	return found, missing, nil
}

func optimizerMetadataAllowlist(version int) []string {
	names := []string{
		"schema_migration",
		"load_job",
		"graph_generation",
		"label_generation",
		"vertex_identity",
		"edge_identity",
		"load_batch",
		"reject_record",
		"deferred_edge",
	}
	if version >= 15 {
		names = append(names, "connector_telemetry")
	}
	if version >= 16 {
		names = append(names, "diagnostic_history")
	}
	if version >= 17 {
		names = append(
			names,
			"job_verification",
			"job_label_counter",
			"load_batch_label_counter",
			"job_unclassified_counter",
		)
	}
	slices.Sort(names)
	return names
}

func validateAnalyzePreconditions(snapshot optimizationSnapshot) error {
	if snapshot.Probe.PostgreSQLStatus != age.ProbePass ||
		snapshot.Probe.AGEPresenceStatus != age.ProbePass ||
		snapshot.Probe.AGEVersionStatus != age.ProbePass ||
		snapshot.Probe.AGELoadabilityStatus != age.ProbePass {
		return errors.New("--apply-analyze requires a compatible, loadable PostgreSQL 17 and Apache AGE 1.6 target")
	}
	if err := snapshot.Schema.RequireCurrent(); err != nil {
		return fmt.Errorf("--apply-analyze requires current metadata: %w", err)
	}
	if !snapshot.GraphAvailable || snapshot.GraphStatus != report.CheckPass {
		return errors.New("--apply-analyze requires an active agefreighter graph generation")
	}
	if snapshot.LabelsTruncated {
		return fmt.Errorf(
			"--apply-analyze refuses more than %d active labels",
			MaxOptimizeLabels,
		)
	}
	if len(snapshot.MetadataRelationsMissing) > 0 {
		return errors.New("--apply-analyze requires the complete current metadata catalog")
	}
	if snapshot.MetadataIndexStatus != report.CheckPass ||
		len(snapshot.RequiredMetadataInvalid) > 0 {
		return errors.New("--apply-analyze requires all version-compatible metadata indexes")
	}
	for _, relation := range append(
		slices.Clone(snapshot.Relations),
		snapshot.MetadataRelations...,
	) {
		if relation.Status != report.CheckPass {
			return errors.New("--apply-analyze requires complete relation catalog evidence")
		}
	}
	return nil
}

func openAnalyzePool(
	ctx context.Context,
	dsn string,
	timeout time.Duration,
) (*pgxpool.Pool, error) {
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, errors.New("parse PostgreSQL target connection for --apply-analyze")
	}
	config.MinConns = 0
	config.MaxConns = 1
	config.ConnConfig.ConnectTimeout = timeout
	config.AfterConnect = nil
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, errors.New("create bounded PostgreSQL --apply-analyze pool")
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, errors.New("connect bounded PostgreSQL --apply-analyze pool")
	}
	return pool, nil
}

func applyBoundedAnalyze(
	ctx context.Context,
	pool *pgxpool.Pool,
	snapshot optimizationSnapshot,
	timeout time.Duration,
) ([]analyzeResult, error) {
	targets := append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	)
	slices.SortFunc(targets, func(left, right relationEvidence) int {
		if compared := strings.Compare(left.Schema, right.Schema); compared != 0 {
			return compared
		}
		return strings.Compare(left.Name, right.Name)
	})
	results := make([]analyzeResult, 0, len(targets))
	for _, target := range targets {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result := analyzeResult{
			Scope:  target.Schema + "." + target.Name,
			Status: "failed",
		}
		operationCtx, cancel := context.WithTimeout(ctx, timeout)
		err := analyzeOneRelation(
			operationCtx,
			pool,
			snapshot.Graph,
			target,
			timeout,
		)
		cancel()
		if err == nil {
			result.Status = "succeeded"
			result.Detail = "catalog identity revalidated; bounded ANALYZE completed"
		} else {
			if parentErr := ctx.Err(); parentErr != nil {
				return nil, parentErr
			}
			result.Detail = safeAnalyzeFailure(err)
		}
		results = append(results, result)
	}
	return results, nil
}

func analyzeOneRelation(
	ctx context.Context,
	pool *pgxpool.Pool,
	graph meta.GraphGeneration,
	target relationEvidence,
	timeout time.Duration,
) error {
	if err := validateAnalyzeTargetAllowlist(target); err != nil {
		return err
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer rollbackAnalyzeTx(tx, timeout)
	lockTimeout := min(timeout, 5*time.Second)
	if _, err := tx.Exec(ctx, `
		SELECT
			pg_catalog.set_config('statement_timeout', $1, true),
			pg_catalog.set_config('lock_timeout', $2, true)`,
		postgresDuration(timeout),
		postgresDuration(lockTimeout),
	); err != nil {
		return err
	}
	store, err := meta.New(tx)
	if err != nil {
		return err
	}
	if _, err := store.LockCurrentSchema(ctx); err != nil {
		return err
	}
	table := pgx.Identifier{target.Schema, target.Name}.Sanitize()
	if _, err := tx.Exec(
		ctx,
		analyzeRelationLockSQL(target.Schema, target.Name),
	); err != nil {
		return err
	}
	if err := validateRelationIdentity(
		ctx,
		tx,
		target.Schema,
		target.Name,
		target.OID,
	); err != nil {
		return err
	}
	if err := validateActiveGraphIdentity(ctx, tx, graph); err != nil {
		return err
	}
	if target.Kind == meta.VertexLabel || target.Kind == meta.EdgeLabel {
		if target.GraphGenerationID != graph.ID {
			return errors.New("AGE label generation ownership changed before operation")
		}
		if err := validateAGELabelIdentity(ctx, tx, target); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(ctx, "ANALYZE "+table); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func analyzeRelationLockSQL(schema, table string) string {
	return "LOCK TABLE " + pgx.Identifier{schema, table}.Sanitize() +
		" IN SHARE UPDATE EXCLUSIVE MODE"
}

func validateAnalyzeTargetAllowlist(target relationEvidence) error {
	if target.Kind == meta.VertexLabel || target.Kind == meta.EdgeLabel {
		if target.Schema == "" || target.Name == "" || target.OID == 0 {
			return errors.New("AGE relation has incomplete optimizer ownership evidence")
		}
		return nil
	}
	if target.Schema != "agefreighter_meta" ||
		target.OID == 0 ||
		!slices.Contains(
			optimizerMetadataAllowlist(meta.SupportedSchemaVersion),
			target.Name,
		) {
		return errors.New("relation is outside the optimizer ANALYZE allowlist")
	}
	return nil
}

func validateActiveGraphIdentity(
	ctx context.Context,
	tx pgx.Tx,
	graph meta.GraphGeneration,
) error {
	var matches bool
	err := tx.QueryRow(ctx, `
		SELECT true
		FROM agefreighter_meta.graph_generation generation
		JOIN ag_catalog.ag_graph graph
		  ON graph.name = generation.graph_name
		 AND graph.graphid = generation.graph_oid
		 AND graph.namespace = generation.namespace_oid
		WHERE generation.graph_generation_id = $1
		  AND generation.graph_name = $2
		  AND generation.graph_oid = $3
		  AND generation.namespace_oid = $4
		  AND generation.state = 'active'
		FOR SHARE OF generation, graph`,
		graph.ID,
		graph.GraphName,
		graph.GraphOID,
		graph.NamespaceOID,
	).Scan(&matches)
	if errors.Is(err, pgx.ErrNoRows) {
		return errors.New("active graph catalog identity changed before operation")
	}
	if err != nil {
		return err
	}
	if !matches {
		return errors.New("active graph catalog identity changed before operation")
	}
	return nil
}

func validateRelationIdentity(
	ctx context.Context,
	tx pgx.Tx,
	schema string,
	name string,
	oid uint32,
) error {
	var matches bool
	err := tx.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1
			FROM pg_catalog.pg_class relation
			JOIN pg_catalog.pg_namespace namespace
			  ON namespace.oid = relation.relnamespace
			WHERE relation.oid = $1
			  AND namespace.nspname = $2
			  AND relation.relname = $3
			  AND relation.relkind IN ('r', 'p')
		)`,
		oid,
		schema,
		name,
	).Scan(&matches)
	if err != nil {
		return err
	}
	if !matches {
		return errors.New("relation catalog identity changed before operation")
	}
	return nil
}

func validateAGELabelIdentity(
	ctx context.Context,
	tx pgx.Tx,
	target relationEvidence,
) error {
	var matches bool
	err := tx.QueryRow(ctx, `
		SELECT true
		FROM agefreighter_meta.label_generation generation
		JOIN ag_catalog.ag_graph graph
		  ON graph.name = $1
		 AND graph.graphid = $5
		 AND graph.namespace = $6
		JOIN ag_catalog.ag_label label
		  ON label.graph = graph.graphid
		 AND label.name = generation.label_name
		 AND label.relation = generation.relation_oid
		 AND label.kind = generation.kind
		 AND label.id = generation.label_id
		JOIN pg_catalog.pg_class sequence
		  ON sequence.oid = generation.sequence_oid
		 AND sequence.relnamespace = graph.namespace
		 AND sequence.relname = label.seq_name
		 AND sequence.relkind = 'S'
		WHERE generation.label_generation_id = $9
		  AND generation.graph_generation_id = $10
		  AND generation.label_name = $2
		  AND generation.kind = $4
		  AND generation.graph_namespace_oid = $6
		  AND generation.label_id = $7
		  AND generation.relation_oid = $3
		  AND generation.sequence_oid = $8
		  AND generation.mapping_generation = $11
		FOR SHARE OF generation, graph, label`,
		target.Schema,
		target.Name,
		target.OID,
		string(target.Kind),
		target.GraphOID,
		target.GraphNamespaceOID,
		target.LabelID,
		target.SequenceOID,
		target.LabelGenerationID,
		target.GraphGenerationID,
		target.MappingGeneration,
	).Scan(&matches)
	if errors.Is(err, pgx.ErrNoRows) {
		return errors.New("AGE label catalog identity changed before operation")
	}
	if err != nil {
		return err
	}
	if !matches {
		return errors.New("AGE label catalog identity changed before operation")
	}
	return nil
}

func rollbackOptimizerTx(tx pgx.Tx) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = tx.Rollback(ctx)
}

func runOptimizerProbe(
	ctx context.Context,
	tx pgx.Tx,
	probe func(pgx.Tx) error,
) error {
	savepoint, err := tx.Begin(ctx)
	if err != nil {
		return fmt.Errorf("%w: %v", errOptimizerSavepointControl, err)
	}
	rollback := func() error {
		rollbackCtx, cancel := context.WithTimeout(
			context.Background(),
			5*time.Second,
		)
		defer cancel()
		if rollbackErr := savepoint.Rollback(rollbackCtx); rollbackErr != nil &&
			!errors.Is(rollbackErr, pgx.ErrTxClosed) {
			return fmt.Errorf(
				"%w: %v",
				errOptimizerSavepointRecovery,
				rollbackErr,
			)
		}
		return nil
	}
	if err := probe(savepoint); err != nil {
		if rollbackErr := rollback(); rollbackErr != nil {
			return rollbackErr
		}
		return err
	}
	if err := savepoint.Commit(ctx); err != nil {
		if rollbackErr := rollback(); rollbackErr != nil {
			return rollbackErr
		}
		return fmt.Errorf("%w: %v", errOptimizerSavepointControl, err)
	}
	return nil
}

func optimizerProbeFatal(ctx context.Context, err error) error {
	if canceledErr := canceled(ctx, err); canceledErr != nil {
		return canceledErr
	}
	if errors.Is(err, errOptimizerSavepointRecovery) ||
		errors.Is(err, errOptimizerSavepointControl) {
		return errors.New("optimizer inspection savepoint failed")
	}
	return nil
}

func rollbackAnalyzeTx(tx pgx.Tx, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), min(timeout, 5*time.Second))
	defer cancel()
	_ = tx.Rollback(ctx)
}

func postgresDuration(value time.Duration) string {
	return fmt.Sprintf("%dms", max(value.Milliseconds(), 1))
}

func optimizerRemaining(ctx context.Context) time.Duration {
	if deadline, ok := ctx.Deadline(); ok {
		return max(time.Until(deadline), time.Millisecond)
	}
	return 30 * time.Second
}

func safeAnalyzeFailure(err error) string {
	var pgErr *pgconn.PgError
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "failed: operation deadline exceeded"
	case errors.Is(err, context.Canceled):
		return "failed: operation canceled"
	case errors.As(err, &pgErr) && pgErr.Code == "42501":
		return "failed: permission denied"
	case errors.As(err, &pgErr) &&
		(pgErr.Code == "57014" || pgErr.Code == "55P03"):
		return "failed: statement or lock timeout"
	case strings.Contains(err.Error(), "catalog identity changed"):
		return "failed: catalog identity changed"
	default:
		return "failed: database operation did not complete"
	}
}

func classifyOptimizerFatal(ctx context.Context, operation string, err error) error {
	if canceledErr := canceled(ctx, err); canceledErr != nil {
		return canceledErr
	}
	return fmt.Errorf("%s: %s", operation, safeDatabaseDetail(err, "database operation failed"))
}

func optimizerEvidenceUnknown(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) &&
		(pgErr.Code == "42501" ||
			pgErr.Code == "57014" ||
			pgErr.Code == "55P03")
}

func buildOptimizationReport(
	snapshot optimizationSnapshot,
	analyze bool,
	at time.Time,
) (report.Document, error) {
	document := report.New("optimize", at)
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
	document.Checks = append(document.Checks, schemaCheck(snapshot.Schema))
	addOptimizationChecks(&document, snapshot, analyze)
	document.Sections = append(
		document.Sections,
		optimizationTargetSection(snapshot, analyze),
		optimizationMigrationSection(snapshot),
		optimizationGraphSection(snapshot),
		optimizationLabelCountSection(snapshot),
		optimizationRelationSection(snapshot),
		optimizationMetadataRelationSection(snapshot),
		optimizationPropertySection(snapshot),
		optimizationIndexSection(snapshot),
		optimizationRecommendationSection(snapshot),
	)
	if analyze {
		document.Sections = append(
			document.Sections,
			optimizationAnalyzeSection(snapshot.AnalyzeResults),
		)
	}
	if snapshot.LabelsTruncated {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "OPTIMIZER_LABELS_TRUNCATED",
			Message: fmt.Sprintf(
				"active label evidence is limited to %d deterministic entries",
				MaxOptimizeLabels,
			),
		})
		document.IncompleteChecks = append(
			document.IncompleteChecks,
			"active-labels",
		)
	}
	if snapshot.BatchAttemptsTruncated {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "BATCH_ATTEMPTS_TRUNCATED",
			Message: fmt.Sprintf(
				"batch-attempt evidence is limited to the first %d ordered attempts",
				MaxOptimizeBatchAttempts,
			),
		})
		document.IncompleteChecks = append(
			document.IncompleteChecks,
			"batch-attempts",
		)
	}
	if indexCatalogTruncated(snapshot) {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "INDEX_CATALOG_TRUNCATED",
			Message: fmt.Sprintf(
				"one or more relation index lists exceeded %d entries",
				MaxOptimizeIndexesPerTable,
			),
		})
		document.IncompleteChecks = append(
			document.IncompleteChecks,
			"index-catalog",
		)
	}
	if indexEvidenceTruncated(snapshot) {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "INDEX_OUTPUT_TRUNCATED",
			Message: fmt.Sprintf(
				"index findings are limited to %d deterministic entries",
				MaxOptimizeIndexFindings,
			),
		})
		document.IncompleteChecks = append(
			document.IncompleteChecks,
			"index-output",
		)
	}
	if recommendationOutputTruncated(snapshot) {
		document.Warnings = append(document.Warnings, report.Finding{
			Code: "RECOMMENDATIONS_TRUNCATED",
			Message: fmt.Sprintf(
				"recommendations are limited to %d deterministic entries",
				MaxOptimizeRecommendations,
			),
		})
		document.IncompleteChecks = append(
			document.IncompleteChecks,
			"recommendations",
		)
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
		return report.Document{}, fmt.Errorf("validate optimizer report: %w", err)
	}
	return document, nil
}

func addOptimizationChecks(
	document *report.Document,
	snapshot optimizationSnapshot,
	analyze bool,
) {
	graphStatus := snapshot.GraphStatus
	graphDetail := snapshot.GraphDetail
	if graphDetail == "" {
		if snapshot.GraphAvailable {
			graphDetail = "one active agefreighter-owned graph generation was selected"
		} else {
			graphDetail = "no active agefreighter-owned graph generation is available"
		}
	}
	addCheck(document, "active-graph", graphStatus,
		"active graph generation ownership was inspected", graphDetail)

	catalogStatus := report.CheckPass
	var catalogProblems int
	var indexesTruncated bool
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		if relation.Status != report.CheckPass {
			catalogStatus = report.CheckUnknown
			catalogProblems++
		}
		indexesTruncated = indexesTruncated || relation.IndexesTruncated
	}
	if snapshot.LabelsTruncated || indexesTruncated {
		catalogStatus = report.CheckUnknown
	}
	addCheck(document, "bounded-catalog-visibility", catalogStatus,
		"bounded graph relation and index catalogs were inspected",
		fmt.Sprintf(
			"labels=%d relation_visibility_problems=%d labels_truncated=%t indexes_truncated=%t",
			len(snapshot.Relations),
			catalogProblems,
			snapshot.LabelsTruncated,
			indexesTruncated,
		))

	metadataStatus := snapshot.MetadataIndexStatus
	if len(snapshot.RequiredMetadataInvalid) > 0 ||
		len(snapshot.MetadataRelationsMissing) > 0 &&
			metadataStatus == report.CheckPass {
		metadataStatus = report.CheckWarning
	}
	addCheck(document, "required-metadata-indexes", metadataStatus,
		"version-compatible agefreighter metadata indexes were inspected",
		fmt.Sprintf(
			"invalid_or_missing_indexes=%d missing_relations=%d schema_version=%d",
			len(snapshot.RequiredMetadataInvalid),
			len(snapshot.MetadataRelationsMissing),
			snapshot.Schema.InstalledVersion,
		))

	ageIndexStatus := report.CheckPass
	var missingAGEIndexes int
	var unknownAGEIndexes int
	for _, relation := range snapshot.Relations {
		missingAGEIndexes += len(relation.RequiredIndexAbsent)
		if relation.RequiredIndexStatus != report.CheckPass {
			unknownAGEIndexes++
		}
	}
	if unknownAGEIndexes > 0 ||
		snapshot.GraphStatus != report.CheckPass ||
		snapshot.LabelsTruncated {
		ageIndexStatus = report.CheckUnknown
	} else if missingAGEIndexes > 0 {
		ageIndexStatus = report.CheckWarning
	}
	addCheck(document, "required-age-indexes", ageIndexStatus,
		"required AGE label indexes were inspected",
		fmt.Sprintf(
			"missing_or_invalid_required_indexes=%d unknown_relations=%d",
			missingAGEIndexes,
			unknownAGEIndexes,
		))

	statsStatus := report.CheckPass
	var neverAnalyzed, stale, unknown int
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		if relation.Status != report.CheckPass {
			unknown++
			continue
		}
		last := latestTime(relation.LastAnalyze, relation.LastAutoAnalyze)
		if last == nil &&
			(relationHasRows(relation) || relation.EstimatedRows < 0) {
			neverAnalyzed++
		}
		if relation.LiveRows > 0 &&
			relation.DeadRows > max(relation.LiveRows/5, 1000) {
			stale++
		}
		if relation.GenerationUpdatedAt != nil &&
			last != nil &&
			last.Before(*relation.GenerationUpdatedAt) {
			stale++
		}
	}
	if unknown > 0 {
		statsStatus = report.CheckUnknown
	} else if neverAnalyzed > 0 || stale > 0 {
		statsStatus = report.CheckWarning
	}
	addCheck(document, "statistics-freshness", statsStatus,
		"bounded analyze and dead-tuple evidence was inspected",
		fmt.Sprintf(
			"never_analyzed=%d stale_indicators=%d unknown=%d",
			neverAnalyzed,
			stale,
			unknown,
		))

	addCheck(document, "property-statistics", report.CheckUnavailable,
		"live AGE property statistics and index recommendations were not produced",
		propertyEvidenceUnavailable)

	duplicateCount, unusedCount := countIndexSignals(snapshot)
	duplicateStatus := report.CheckPass
	if duplicateCount > 0 {
		duplicateStatus = report.CheckWarning
	}
	addCheck(document, "duplicate-indexes", duplicateStatus,
		"exact catalog-equivalent indexes were inspected",
		fmt.Sprintf("duplicate_groups=%d", duplicateCount))
	usageStatus := snapshot.StatsResetStatus
	if usageStatus == report.CheckPass && unusedCount > 0 {
		usageStatus = report.CheckWarning
	}
	addCheck(document, "index-usage", usageStatus,
		"bounded index usage was inspected with its statistics-reset timestamp",
		fmt.Sprintf("zero_scan_indexes=%d", unusedCount))

	visibilityStatus := report.CheckPass
	if snapshot.DatabaseSizeStatus != report.CheckPass ||
		snapshot.WALStatus != report.CheckPass {
		visibilityStatus = report.CheckUnknown
	}
	addCheck(document, "storage-wal-visibility", visibilityStatus,
		"database storage and WAL counters were inspected",
		"filesystem free capacity is not exposed by portable PostgreSQL catalogs")

	if analyze {
		failed := 0
		for _, result := range snapshot.AnalyzeResults {
			if result.Status != "succeeded" {
				failed++
			}
		}
		status := report.CheckPass
		if failed > 0 {
			status = report.CheckUnknown
		}
		addCheck(document, "analyze", status,
			"explicitly requested bounded ANALYZE operations were reported",
			fmt.Sprintf(
				"attempted=%d succeeded=%d failed=%d",
				len(snapshot.AnalyzeResults),
				len(snapshot.AnalyzeResults)-failed,
				failed,
			))
	}
}

func optimizationTargetSection(
	snapshot optimizationSnapshot,
	analyze bool,
) report.Section {
	mode := "recommendation-only"
	if analyze {
		mode = "explicit-analyze"
	}
	fields := []report.Field{
		passField("mode", mode),
		passField("evidencePhase", "captured before any explicitly requested ANALYZE"),
		{
			Name: "metadataInstalledVersion",
			Value: func() string {
				if snapshot.Schema.State == meta.SchemaUnknown {
					return "unknown"
				}
				return strconv.Itoa(snapshot.Schema.InstalledVersion)
			}(),
			Status: func() report.CheckStatus {
				if snapshot.Schema.State == meta.SchemaUnknown {
					return report.CheckUnknown
				}
				return report.CheckPass
			}(),
		},
		passField("metadataSupportedVersion", strconv.Itoa(snapshot.Schema.SupportedVersion)),
		report.Field{
			Name: "databaseBytes", Value: strconv.FormatInt(snapshot.DatabaseBytes, 10),
			Status: snapshot.DatabaseSizeStatus,
		},
		report.Field{
			Name: "walBytesSinceReset", Value: strconv.FormatInt(snapshot.WALBytes, 10),
			Status: snapshot.WALStatus,
		},
		report.Field{
			Name:   "walStatisticsReset",
			Value:  optionalTimestamp(snapshot.WALReset),
			Status: optionalVisibilityStatus(snapshot.WALStatus, snapshot.WALReset),
		},
		report.Field{
			Name:   "databaseStatisticsReset",
			Value:  optionalTimestamp(snapshot.StatsReset),
			Status: optionalVisibilityStatus(snapshot.StatsResetStatus, snapshot.StatsReset),
		},
		{
			Name:   "filesystemFreeBytes",
			Value:  "unknown; use platform storage monitoring",
			Status: report.CheckUnavailable,
		},
		{
			Name: "ginAgtypeOperatorClass",
			Value: func() string {
				if snapshot.GINStatus != report.CheckPass {
					return "operator-class visibility is unknown"
				}
				if snapshot.GINSupported {
					return "supported allowlisted AGE 1.6 operator class detected"
				}
				return "no supported allowlisted AGE 1.6 operator class detected"
			}(),
			Status: snapshot.GINStatus,
		},
	}
	return report.Section{Title: "Optimizer target evidence", Fields: fields}
}

func optimizationMigrationSection(snapshot optimizationSnapshot) report.Section {
	if !snapshot.JobAvailable {
		status := snapshot.MigrationStatus
		detail := snapshot.MigrationDetail
		if detail == "" {
			detail = "migration metadata is unavailable"
		}
		return report.Section{
			Title: "Migration counters and telemetry",
			Fields: []report.Field{{
				Name: "activeGenerationJob", Value: detail, Status: status,
			}},
		}
	}
	fields := []report.Field{
		passField("status", string(snapshot.Job.Status)),
		passField("sourceType", snapshot.Job.SourceType),
		passField("loadMode", snapshot.Job.LoadMode),
		passField("committedRows", strconv.FormatInt(snapshot.Job.CommittedRows, 10)),
		passField("committedBytes", strconv.FormatInt(snapshot.Job.CommittedBytes, 10)),
		passField("rejectedRows", strconv.FormatInt(snapshot.Job.RejectedRows, 10)),
		passField("sourceRejectedRows", strconv.FormatInt(snapshot.Job.SourceRejectedRows, 10)),
		passField("committedBatchCount", strconv.FormatUint(snapshot.Job.NextBatchID-1, 10)),
		{
			Name:   "batchAttemptsObserved",
			Value:  strconv.Itoa(snapshot.BatchAttemptsObserved),
			Status: snapshot.BatchAttemptsStatus,
		},
		{
			Name:   "batchAttemptsTruncated",
			Value:  strconv.FormatBool(snapshot.BatchAttemptsTruncated),
			Status: snapshot.BatchAttemptsStatus,
		},
	}
	if snapshot.MigrationStatus == report.CheckUnknown {
		fields = append(fields, report.Field{
			Name:   "visibility",
			Value:  snapshot.MigrationDetail,
			Status: report.CheckUnknown,
		})
	}
	if snapshot.LatestBatchAvailable {
		fields = append(fields,
			passField("latestBatchStatus", string(snapshot.LatestBatch.Status)),
			passField("latestBatchAttempt", strconv.FormatUint(
				uint64(snapshot.LatestBatch.Attempt),
				10,
			)),
			passField("latestBatchRows", strconv.FormatInt(snapshot.LatestBatch.Rows, 10)),
			passField("latestBatchBytes", strconv.FormatInt(snapshot.LatestBatch.Bytes, 10)),
			passField(
				"latestBatchRejectedRows",
				strconv.FormatInt(snapshot.LatestBatch.RejectedRows, 10),
			),
		)
	} else {
		fields = append(fields, unavailableField(
			"latestBatch",
			"no batch telemetry was recorded",
		))
	}
	if snapshot.TelemetryAvailable {
		fields = append(fields,
			passField("connectorPages", strconv.FormatInt(snapshot.Telemetry.Pages, 10)),
			passField(
				"failedRequestAttempts",
				strconv.FormatInt(snapshot.Telemetry.FailedRequestAttempts, 10),
			),
			passField(
				"throttledRequests",
				strconv.FormatInt(snapshot.Telemetry.ThrottledRequests, 10),
			),
			passField(
				"requestCharge",
				strconv.FormatFloat(snapshot.Telemetry.RequestCharge, 'f', -1, 64),
			),
		)
	} else {
		fields = append(fields, unavailableField(
			"connectorTelemetry",
			"not recorded or unavailable in this metadata version",
		))
	}
	if snapshot.CountersAvailable {
		var complete, incomplete int
		var rows, bytes, rejected int64
		for _, counter := range snapshot.Counters {
			if counter.Completeness != meta.CounterComplete ||
				counter.CommittedRows == nil ||
				counter.RejectedRows == nil {
				incomplete++
				continue
			}
			complete++
			rows += *counter.CommittedRows
			rejected += *counter.RejectedRows
			if counter.CommittedBytes != nil {
				bytes += *counter.CommittedBytes
			}
		}
		fields = append(fields,
			passField("labelCountersComplete", strconv.Itoa(complete)),
			passField("labelCountersIncomplete", strconv.Itoa(incomplete)),
			passField("labelCounterCommittedRows", strconv.FormatInt(rows, 10)),
			passField("labelCounterCommittedBytesKnown", strconv.FormatInt(bytes, 10)),
			passField("labelCounterRejectedRows", strconv.FormatInt(rejected, 10)),
		)
	} else {
		fields = append(fields, unavailableField(
			"labelCounters",
			"per-label counters require metadata schema v17",
		))
	}
	return report.Section{Title: "Migration counters and telemetry", Fields: fields}
}

func optimizationGraphSection(snapshot optimizationSnapshot) report.Section {
	if !snapshot.GraphAvailable {
		return unavailableSection(
			"Graph size and density",
			"graph",
			"no active graph generation is available",
		)
	}
	var vertexRows, edgeRows float64
	aggregatesComplete := snapshot.GraphStatus == report.CheckPass &&
		!snapshot.LabelsTruncated &&
		len(snapshot.Relations) == len(snapshot.Labels)
	rowEstimatesKnown := aggregatesComplete
	var graphBytes, indexBytes int64
	for _, relation := range snapshot.Relations {
		if relation.Status != report.CheckPass {
			aggregatesComplete = false
			rowEstimatesKnown = false
			continue
		}
		graphBytes += relation.TotalBytes
		indexBytes += relation.IndexBytes
		switch relation.Kind {
		case meta.VertexLabel:
			if relation.EstimatedRows < 0 {
				rowEstimatesKnown = false
			} else {
				vertexRows += relation.EstimatedRows
			}
		case meta.EdgeLabel:
			if relation.EstimatedRows < 0 {
				rowEstimatesKnown = false
			} else {
				edgeRows += relation.EstimatedRows
			}
		default:
			aggregatesComplete = false
			rowEstimatesKnown = false
		}
	}
	density := "unknown"
	densityStatus := report.CheckUnavailable
	rowStatus := report.CheckPass
	if !rowEstimatesKnown {
		rowStatus = report.CheckUnknown
	}
	if rowEstimatesKnown && vertexRows > 0 {
		density = strconv.FormatFloat(edgeRows/vertexRows, 'f', 6, 64)
		densityStatus = report.CheckPass
	}
	aggregateStatus := report.CheckPass
	graphBytesValue := strconv.FormatInt(graphBytes, 10)
	indexBytesValue := strconv.FormatInt(indexBytes, 10)
	if !aggregatesComplete {
		aggregateStatus = report.CheckUnknown
		graphBytesValue = "unknown; contributing label evidence is incomplete"
		indexBytesValue = "unknown; contributing label evidence is incomplete"
		density = "unknown; contributing label evidence is incomplete"
		densityStatus = report.CheckUnknown
	}
	return report.Section{Title: "Graph size and density", Fields: []report.Field{
		{
			Name: "activeLabelsInspected", Value: strconv.Itoa(len(snapshot.Relations)),
			Status: aggregateStatus,
		},
		{
			Name: "estimatedVertexRows", Value: estimatedRowsValue(vertexRows, rowEstimatesKnown),
			Status: rowStatus,
		},
		{
			Name: "estimatedEdgeRows", Value: estimatedRowsValue(edgeRows, rowEstimatesKnown),
			Status: rowStatus,
		},
		{Name: "estimatedEdgeDensity", Value: density, Status: densityStatus},
		{Name: "labelRelationBytes", Value: graphBytesValue, Status: aggregateStatus},
		{Name: "labelIndexBytes", Value: indexBytesValue, Status: aggregateStatus},
	}}
}

func optimizationLabelCountSection(snapshot optimizationSnapshot) report.Section {
	section := report.Section{
		Title:  "Per-label row evidence",
		Fields: []report.Field{},
	}
	counters := make(map[int64]meta.LabelCounter, len(snapshot.Counters))
	for _, counter := range snapshot.Counters {
		counters[counter.LabelGenerationID] = counter
	}
	relations := make(map[string]relationEvidence, len(snapshot.Relations))
	for _, relation := range snapshot.Relations {
		relations[string(relation.Kind)+"\x00"+relation.Name] = relation
	}
	for index, label := range snapshot.Labels {
		relation := relations[string(label.Kind)+"\x00"+label.LabelName]
		value := fmt.Sprintf(
			"kind=%s catalogEstimatedRows=%s statisticsLiveRows=%d",
			string(label.Kind),
			estimatedRowsValue(relation.EstimatedRows, relation.EstimatedRows >= 0),
			relation.LiveRows,
		)
		status := relation.Status
		counter, exists := counters[label.ID]
		if !snapshot.CountersAvailable || !exists {
			value += " storedCommittedRows=unavailable counterCompleteness=unavailable"
			status = report.CheckUnavailable
		} else if counter.Completeness != meta.CounterComplete ||
			counter.CommittedRows == nil {
			value += " storedCommittedRows=unknown counterCompleteness=incomplete"
			status = report.CheckUnknown
		} else {
			value += " storedCommittedRows=" +
				strconv.FormatInt(*counter.CommittedRows, 10) +
				" counterCompleteness=complete"
		}
		section.Fields = append(section.Fields, report.Field{
			Name: fmt.Sprintf(
				"%03d.%s.%s",
				index+1,
				relationKindName(label.Kind),
				safeFieldName(label.LabelName),
			),
			Value:  value,
			Status: status,
		})
	}
	if len(section.Fields) == 0 {
		section.Fields = append(section.Fields, unavailableField(
			"labels",
			"no active label row evidence is available",
		))
	}
	return section
}

func optimizationRelationSection(snapshot optimizationSnapshot) report.Section {
	section := report.Section{Title: "Bounded relation statistics", Fields: []report.Field{}}
	for index, relation := range snapshot.Relations {
		status := relation.Status
		value := relation.Detail
		if status == report.CheckPass {
			value = fmt.Sprintf(
				"kind=%s estimatedRows=%s liveRows=%d deadRows=%d totalBytes=%d indexBytes=%d seqScans=%d indexScans=%d lastAnalyze=%s lastAutoAnalyze=%s",
				string(relation.Kind),
				estimatedRowsValue(relation.EstimatedRows, relation.EstimatedRows >= 0),
				relation.LiveRows,
				relation.DeadRows,
				relation.TotalBytes,
				relation.IndexBytes,
				relation.SequentialScans,
				relation.IndexScans,
				optionalTimestamp(relation.LastAnalyze),
				optionalTimestamp(relation.LastAutoAnalyze),
			)
		}
		section.Fields = append(section.Fields, report.Field{
			Name: fmt.Sprintf(
				"%03d.%s.%s",
				index+1,
				relationKindName(relation.Kind),
				safeFieldName(relation.Name),
			),
			Value:  boundedReportValue(value),
			Status: status,
		})
	}
	if len(section.Fields) == 0 {
		section.Fields = append(section.Fields, unavailableField(
			"labels",
			"no active label relations were available",
		))
	}
	return section
}

func optimizationMetadataRelationSection(snapshot optimizationSnapshot) report.Section {
	section := report.Section{
		Title:  "Metadata relation statistics",
		Fields: []report.Field{},
	}
	for index, relation := range snapshot.MetadataRelations {
		status := relation.Status
		value := relation.Detail
		if status == report.CheckPass {
			value = fmt.Sprintf(
				"estimatedRows=%s liveRows=%d deadRows=%d totalBytes=%d indexBytes=%d seqScans=%d indexScans=%d lastAnalyze=%s lastAutoAnalyze=%s",
				estimatedRowsValue(relation.EstimatedRows, relation.EstimatedRows >= 0),
				relation.LiveRows,
				relation.DeadRows,
				relation.TotalBytes,
				relation.IndexBytes,
				relation.SequentialScans,
				relation.IndexScans,
				optionalTimestamp(relation.LastAnalyze),
				optionalTimestamp(relation.LastAutoAnalyze),
			)
		}
		section.Fields = append(section.Fields, report.Field{
			Name:   fmt.Sprintf("%03d.%s", index+1, safeFieldName(relation.Name)),
			Value:  boundedReportValue(value),
			Status: status,
		})
	}
	for _, name := range snapshot.MetadataRelationsMissing {
		section.Fields = append(section.Fields, report.Field{
			Name:   "missing." + safeFieldName(name),
			Value:  "allowlisted relation is absent",
			Status: report.CheckUnavailable,
		})
	}
	return section
}

func optimizationPropertySection(_ optimizationSnapshot) report.Section {
	return report.Section{
		Title: "AGE property evidence",
		Fields: []report.Field{{
			Name:   "cardinalityAndIndexRecommendations",
			Value:  propertyEvidenceUnavailable,
			Status: report.CheckUnavailable,
		}},
	}
}

func optimizationIndexSection(snapshot optimizationSnapshot) report.Section {
	section := report.Section{Title: "Index evidence", Fields: []report.Field{}}
	appendFinding := func(field report.Field) {
		if len(section.Fields) < MaxOptimizeIndexFindings {
			section.Fields = append(section.Fields, field)
		}
	}
	duplicateNumber := 0
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		groups := duplicateIndexGroups(relation.Indexes)
		for _, group := range groups {
			duplicateNumber++
			appendFinding(report.Field{
				Name: fmt.Sprintf("duplicate.%03d", duplicateNumber),
				Value: fmt.Sprintf(
					"scope=%s.%s exactEquivalentIndexes=%s evidence=catalog structural signature",
					relation.Schema,
					relation.Name,
					strings.Join(group, ","),
				),
				Status: report.CheckWarning,
			})
		}
	}
	missingNumber := 0
	for _, relation := range snapshot.Relations {
		for _, column := range relation.RequiredIndexAbsent {
			missingNumber++
			appendFinding(report.Field{
				Name: fmt.Sprintf("requiredAgeIndex.%03d", missingNumber),
				Value: fmt.Sprintf(
					"scope=%s.%s missingOrInvalidExactDefinition=%s",
					relation.Schema,
					relation.Name,
					column,
				),
				Status: report.CheckWarning,
			})
		}
	}
	for index, name := range snapshot.RequiredMetadataInvalid {
		appendFinding(report.Field{
			Name:   fmt.Sprintf("requiredMetadataIndex.%03d", index+1),
			Value:  "missing or structurally invalid: " + name,
			Status: report.CheckWarning,
		})
	}
	unusedNumber := 0
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		for _, index := range relation.Indexes {
			if !index.Valid || !index.Ready || index.Primary || index.Scans != 0 {
				continue
			}
			unusedNumber++
			appendFinding(report.Field{
				Name: fmt.Sprintf("zeroScan.%03d", unusedNumber),
				Value: fmt.Sprintf(
					"scope=%s.%s index=%s scans=0 statisticsReset=%s evidence=bounded-pg-stat-all-indexes",
					relation.Schema,
					relation.Name,
					index.Name,
					optionalTimestamp(snapshot.StatsReset),
				),
				Status: report.CheckWarning,
			})
		}
	}
	if len(section.Fields) == 0 {
		section.Fields = append(section.Fields, passField(
			"summary",
			"no exact duplicate or missing required indexes were detected in the bounded catalog",
		))
	}
	return section
}

func optimizationRecommendationSection(
	snapshot optimizationSnapshot,
) report.Section {
	section := report.Section{Title: "Recommendations", Fields: []report.Field{}}
	number := 0
	add := func(value string, status report.CheckStatus) {
		number++
		if len(section.Fields) >= MaxOptimizeRecommendations {
			return
		}
		section.Fields = append(section.Fields, report.Field{
			Name:   fmt.Sprintf("%03d", number),
			Value:  boundedReportValue(value),
			Status: status,
		})
	}
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		lastAnalyze := latestTime(relation.LastAnalyze, relation.LastAutoAnalyze)
		if relation.Status == report.CheckPass &&
			(relationHasRows(relation) || relation.EstimatedRows < 0) &&
			(lastAnalyze == nil ||
				relation.DeadRows > max(relation.LiveRows/5, 1000) ||
				relation.GenerationUpdatedAt != nil &&
					lastAnalyze.Before(*relation.GenerationUpdatedAt)) {
			add(fmt.Sprintf(
				"action=ANALYZE scope=%s.%s confidence=high evidence=missing-or-stale-statistics sql=%s",
				relation.Schema,
				relation.Name,
				"ANALYZE "+pgx.Identifier{relation.Schema, relation.Name}.Sanitize()+";",
			), report.CheckWarning)
		}
	}
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		for _, group := range duplicateIndexGroups(relation.Indexes) {
			add(fmt.Sprintf(
				"action=review-exact-duplicate-indexes scope=%s.%s confidence=high evidence=catalog-equivalent-definitions indexes=%s limitation=no-drop-sql-emitted",
				relation.Schema,
				relation.Name,
				strings.Join(group, ","),
			), report.CheckWarning)
		}
	}
	if number == 0 {
		section.Fields = append(section.Fields, passField(
			"summary",
			"no recommendation met the bounded evidence rules",
		))
	}
	return section
}

func optimizationAnalyzeSection(results []analyzeResult) report.Section {
	section := report.Section{Title: "ANALYZE operations", Fields: []report.Field{}}
	for index, result := range results {
		status := report.CheckPass
		if result.Status != "succeeded" {
			status = report.CheckUnknown
		}
		section.Fields = append(section.Fields, report.Field{
			Name: fmt.Sprintf("%03d.%s", index+1, safeFieldName(result.Scope)),
			Value: fmt.Sprintf(
				"attempted=true status=%s detail=%s",
				result.Status,
				result.Detail,
			),
			Status: status,
		})
	}
	if len(results) == 0 {
		section.Fields = append(section.Fields, passField(
			"summary",
			"no allowlisted relations were available",
		))
	}
	return section
}

func duplicateIndexGroups(indexes []indexEvidence) [][]string {
	bySignature := make(map[string][]string)
	for _, index := range indexes {
		if !index.Valid || !index.Ready {
			continue
		}
		bySignature[index.Signature] = append(
			bySignature[index.Signature],
			index.Name,
		)
	}
	groups := make([][]string, 0)
	for _, names := range bySignature {
		if len(names) < 2 {
			continue
		}
		slices.Sort(names)
		groups = append(groups, names)
	}
	slices.SortFunc(groups, func(left, right []string) int {
		return strings.Compare(strings.Join(left, "\x00"), strings.Join(right, "\x00"))
	})
	return groups
}

func countIndexSignals(snapshot optimizationSnapshot) (int, int) {
	var duplicate, unused int
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		duplicate += len(duplicateIndexGroups(relation.Indexes))
		for _, index := range relation.Indexes {
			if index.Valid && index.Ready && !index.Primary && index.Scans == 0 {
				unused++
			}
		}
	}
	return duplicate, unused
}

func indexEvidenceTruncated(snapshot optimizationSnapshot) bool {
	count := len(snapshot.RequiredMetadataInvalid)
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		count += len(duplicateIndexGroups(relation.Indexes))
		for _, index := range relation.Indexes {
			if index.Valid && index.Ready && !index.Primary && index.Scans == 0 {
				count++
			}
		}
	}
	for _, relation := range snapshot.Relations {
		count += len(relation.RequiredIndexAbsent)
	}
	return count > MaxOptimizeIndexFindings
}

func indexCatalogTruncated(snapshot optimizationSnapshot) bool {
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		if relation.IndexesTruncated {
			return true
		}
	}
	return false
}

func recommendationOutputTruncated(snapshot optimizationSnapshot) bool {
	count := 0
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		lastAnalyze := latestTime(relation.LastAnalyze, relation.LastAutoAnalyze)
		if relation.Status == report.CheckPass &&
			(relationHasRows(relation) || relation.EstimatedRows < 0) &&
			(lastAnalyze == nil ||
				relation.DeadRows > max(relation.LiveRows/5, 1000) ||
				relation.GenerationUpdatedAt != nil &&
					lastAnalyze.Before(*relation.GenerationUpdatedAt)) {
			count++
		}
	}
	for _, relation := range append(
		slices.Clone(snapshot.MetadataRelations),
		snapshot.Relations...,
	) {
		count += len(duplicateIndexGroups(relation.Indexes))
	}
	return count > MaxOptimizeRecommendations
}

func latestTime(values ...*time.Time) *time.Time {
	var latest *time.Time
	for _, value := range values {
		if value != nil && (latest == nil || value.After(*latest)) {
			latest = value
		}
	}
	return latest
}

func optionalTimestamp(value *time.Time) string {
	if value == nil {
		return "unknown"
	}
	return formatTime(*value)
}

func relationHasRows(relation relationEvidence) bool {
	return relation.LiveRows > 0 ||
		relation.EstimatedRows > 0
}

func estimatedRowsValue(value float64, known bool) string {
	if !known {
		return "unknown"
	}
	return strconv.FormatFloat(value, 'f', 0, 64)
}

func optionalVisibilityStatus(
	visibility report.CheckStatus,
	value *time.Time,
) report.CheckStatus {
	if visibility != report.CheckPass {
		return visibility
	}
	if value == nil {
		return report.CheckUnavailable
	}
	return report.CheckPass
}

func relationKindName(kind meta.LabelKind) string {
	switch kind {
	case meta.VertexLabel:
		return "vertex"
	case meta.EdgeLabel:
		return "edge"
	default:
		return "relation"
	}
}

func safeFieldName(value string) string {
	value = strings.ReplaceAll(value, "\x00", "\uFFFD")
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	if len(value) <= 256 {
		return value
	}
	value = value[:253]
	for !utf8.ValidString(value) {
		value = value[:len(value)-1]
	}
	return value + "..."
}
