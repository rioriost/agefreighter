package meta

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

const MaxContinuationDigestBytes = 128

type ConnectorTelemetry struct {
	JobID                 string    `json:"jobId"`
	Connector             string    `json:"connector"`
	Pages                 int64     `json:"pages"`
	RequestCharge         float64   `json:"requestCharge"`
	FailedRequestAttempts int64     `json:"failedRequestAttempts"`
	ThrottledRequests     int64     `json:"throttledRequests"`
	ContinuationDigest    string    `json:"continuationDigest,omitempty"`
	RecordedAt            time.Time `json:"recordedAt"`
}

func validateConnectorTelemetry(value ConnectorTelemetry) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	switch value.Connector {
	case "csv", "postgresql", "neo4j", "cosmos-nosql":
	default:
		return fmt.Errorf("unsupported telemetry connector %q", value.Connector)
	}
	if value.Pages < 0 || value.RequestCharge < 0 ||
		math.IsNaN(value.RequestCharge) || math.IsInf(value.RequestCharge, 0) ||
		value.FailedRequestAttempts < 0 || value.ThrottledRequests < 0 {
		return errors.New("connector telemetry counters must be finite and non-negative")
	}
	if len(value.ContinuationDigest) > MaxContinuationDigestBytes {
		return fmt.Errorf(
			"connector continuation digest exceeds %d bytes",
			MaxContinuationDigestBytes,
		)
	}
	if strings.ContainsAny(value.ContinuationDigest, "\r\n") {
		return errors.New("connector continuation digest must be a single line")
	}
	if value.ContinuationDigest != "" {
		if len(value.ContinuationDigest) < 8 ||
			value.ContinuationDigest != strings.ToLower(value.ContinuationDigest) {
			return errors.New("connector continuation digest must be lowercase hexadecimal")
		}
		if _, err := hex.DecodeString(value.ContinuationDigest); err != nil {
			return errors.New("connector continuation digest must be lowercase hexadecimal")
		}
	}
	return nil
}

// PutConnectorTelemetry stores one bounded, non-secret summary for a job.
// Replaying an identical summary is idempotent; conflicting summaries fail.
func (store *Store) PutConnectorTelemetry(
	ctx context.Context,
	value ConnectorTelemetry,
) error {
	if err := validateConnectorTelemetry(value); err != nil {
		return err
	}
	tag, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.connector_telemetry (
			job_id, connector, pages, request_charge,
			failed_request_attempts, throttled_requests, continuation_digest
		) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (job_id) DO NOTHING`,
		value.JobID,
		value.Connector,
		value.Pages,
		value.RequestCharge,
		value.FailedRequestAttempts,
		value.ThrottledRequests,
		value.ContinuationDigest,
	)
	if err != nil {
		return fmt.Errorf("store connector telemetry: %w", err)
	}
	if tag.RowsAffected() == 1 {
		return nil
	}
	var identical bool
	err = store.database.QueryRow(
		ctx,
		`SELECT
			connector = $2
			AND pages = $3
			AND request_charge = $4
			AND failed_request_attempts = $5
			AND throttled_requests = $6
			AND continuation_digest = $7
		 FROM agefreighter_meta.connector_telemetry
		 WHERE job_id = $1::uuid`,
		value.JobID,
		value.Connector,
		value.Pages,
		value.RequestCharge,
		value.FailedRequestAttempts,
		value.ThrottledRequests,
		value.ContinuationDigest,
	).Scan(&identical)
	if err != nil {
		return fmt.Errorf("compare connector telemetry replay: %w", err)
	}
	if !identical {
		return fmt.Errorf("%w: connector telemetry already differs", ErrConflict)
	}
	return nil
}

func (store *Store) GetConnectorTelemetry(
	ctx context.Context,
	jobID string,
) (ConnectorTelemetry, error) {
	if err := validateJobID(jobID); err != nil {
		return ConnectorTelemetry{}, err
	}
	var value ConnectorTelemetry
	err := store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, connector, pages, request_charge,
			failed_request_attempts, throttled_requests,
			continuation_digest, recorded_at
		 FROM agefreighter_meta.connector_telemetry
		 WHERE job_id = $1::uuid`,
		jobID,
	).Scan(
		&value.JobID,
		&value.Connector,
		&value.Pages,
		&value.RequestCharge,
		&value.FailedRequestAttempts,
		&value.ThrottledRequests,
		&value.ContinuationDigest,
		&value.RecordedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ConnectorTelemetry{}, fmt.Errorf(
			"%w: connector telemetry for job %q",
			ErrNotFound,
			jobID,
		)
	}
	if err != nil {
		return ConnectorTelemetry{}, fmt.Errorf("read connector telemetry: %w", err)
	}
	return value, nil
}
