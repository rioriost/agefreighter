package age

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type ProbeStatus string

const (
	ProbePass        ProbeStatus = "pass"
	ProbeFail        ProbeStatus = "fail"
	ProbeUnknown     ProbeStatus = "unknown"
	ProbeUnavailable ProbeStatus = "unavailable"
)

type DegradedProbe struct {
	PostgreSQLVersionNumber int           `json:"postgresqlVersionNumber,omitempty"`
	PostgreSQLVersion       string        `json:"postgresqlVersion,omitempty"`
	PostgreSQLMajor         int           `json:"postgresqlMajor,omitempty"`
	PostgreSQLStatus        ProbeStatus   `json:"postgresqlStatus"`
	PostgreSQLDetail        string        `json:"postgresqlDetail,omitempty"`
	AGEPresenceStatus       ProbeStatus   `json:"agePresenceStatus"`
	AGEPresenceDetail       string        `json:"agePresenceDetail,omitempty"`
	AGEVersion              string        `json:"ageVersion,omitempty"`
	AGEVersionStatus        ProbeStatus   `json:"ageVersionStatus"`
	AGEVersionDetail        string        `json:"ageVersionDetail,omitempty"`
	AGELoadabilityStatus    ProbeStatus   `json:"ageLoadabilityStatus"`
	AGELoadabilityDetail    string        `json:"ageLoadabilityDetail,omitempty"`
	AGEPreloadStatus        PreloadStatus `json:"agePreloadStatus"`
}

type ProbeOptions struct {
	ConnectTimeout   time.Duration
	OperationTimeout time.Duration
}

// ProbeDegraded connects as plain PostgreSQL. It intentionally has no
// AfterConnect hook and executes only read-only statements.
func ProbeDegraded(
	ctx context.Context,
	dsn string,
	options ProbeOptions,
) (DegradedProbe, error) {
	if strings.TrimSpace(dsn) == "" {
		return DegradedProbe{}, errors.New("PostgreSQL connection string is required")
	}
	if options.ConnectTimeout <= 0 {
		return DegradedProbe{}, errors.New("connect timeout must be positive")
	}
	if options.OperationTimeout <= 0 {
		return DegradedProbe{}, errors.New("operation timeout must be positive")
	}
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return DegradedProbe{}, fmt.Errorf("parse PostgreSQL connection string: %w", err)
	}
	config.MinConns = 0
	config.MaxConns = 1
	config.ConnConfig.ConnectTimeout = options.ConnectTimeout
	config.AfterConnect = nil
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return DegradedProbe{}, fmt.Errorf("create PostgreSQL probe pool: %w", err)
	}
	defer pool.Close()
	pingCtx, cancel := context.WithTimeout(ctx, options.OperationTimeout)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		return DegradedProbe{}, fmt.Errorf("connect to PostgreSQL: %w", err)
	}
	probeCtx, probeCancel := context.WithTimeout(ctx, options.OperationTimeout)
	defer probeCancel()
	return probeDegradedCapabilities(probeCtx, pool)
}

type degradedProbeDatabase interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}

func probeDegradedCapabilities(
	ctx context.Context,
	database degradedProbeDatabase,
) (DegradedProbe, error) {
	result := DegradedProbe{
		PostgreSQLStatus:     ProbeUnknown,
		AGEPresenceStatus:    ProbeUnknown,
		AGEVersionStatus:     ProbeUnknown,
		AGELoadabilityStatus: ProbeUnknown,
		AGEPreloadStatus:     PreloadUnknown,
	}

	var versionNumberText string
	err := database.QueryRow(
		ctx,
		`SELECT current_setting('server_version_num'), current_setting('server_version')`,
	).Scan(&versionNumberText, &result.PostgreSQLVersion)
	if err != nil {
		if isInsufficientPrivilege(err) {
			result.PostgreSQLDetail = "permission denied while reading PostgreSQL version"
		} else {
			return DegradedProbe{}, fmt.Errorf("probe PostgreSQL version: %w", err)
		}
	} else {
		versionNumber, parseErr := strconv.Atoi(versionNumberText)
		if parseErr != nil || versionNumber <= 0 {
			result.PostgreSQLDetail = fmt.Sprintf(
				"server_version_num %q is not valid",
				versionNumberText,
			)
		} else {
			result.PostgreSQLVersionNumber = versionNumber
			result.PostgreSQLMajor = versionNumber / 10000
			if isSupportedPostgreSQLMajor(result.PostgreSQLMajor) {
				result.PostgreSQLStatus = ProbePass
			} else {
				result.PostgreSQLStatus = ProbeFail
				result.PostgreSQLDetail = fmt.Sprintf(
					"unsupported PostgreSQL major version %d; supported majors are 14 through 18",
					result.PostgreSQLMajor,
				)
			}
		}
	}

	var ageVersionText string
	err = database.QueryRow(
		ctx,
		`SELECT extversion
		 FROM pg_catalog.pg_extension
		 WHERE extname = 'age'`,
	).Scan(&ageVersionText)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		result.AGEPresenceStatus = ProbeUnavailable
		result.AGEPresenceDetail = "Apache AGE extension is not installed"
		result.AGEVersionStatus = ProbeUnavailable
		result.AGELoadabilityStatus = ProbeUnavailable
		return result, nil
	case isInsufficientPrivilege(err):
		result.AGEPresenceDetail = "permission denied while inspecting installed extensions"
		return result, nil
	case err != nil:
		return DegradedProbe{}, fmt.Errorf("probe Apache AGE extension: %w", err)
	default:
		result.AGEPresenceStatus = ProbePass
		result.AGEVersion = ageVersionText
	}

	ageVersion, err := ParseVersion(ageVersionText)
	if err != nil {
		result.AGEVersionStatus = ProbeUnknown
		result.AGEVersionDetail = fmt.Sprintf(
			"Apache AGE extension version %q is not valid",
			ageVersionText,
		)
	} else if result.PostgreSQLVersionNumber > 0 &&
		isSupportedTargetVersion(result.PostgreSQLMajor, ageVersion) {
		result.AGEVersionStatus = ProbePass
	} else if result.PostgreSQLVersionNumber == 0 &&
		isSupportedAGEVersion(ageVersion) {
		result.AGEVersionStatus = ProbePass
	} else {
		result.AGEVersionStatus = ProbeFail
		if result.PostgreSQLVersionNumber > 0 &&
			isSupportedPostgreSQLMajor(result.PostgreSQLMajor) {
			result.AGEVersionDetail = fmt.Sprintf(
				"unsupported Apache AGE version %s for PostgreSQL %d; supported series is %s",
				ageVersion,
				result.PostgreSQLMajor,
				supportedAGESeries(result.PostgreSQLMajor),
			)
		} else {
			result.AGEVersionDetail = fmt.Sprintf(
				"Apache AGE version %s cannot be paired with unsupported PostgreSQL major %d",
				ageVersion,
				result.PostgreSQLMajor,
			)
		}
	}

	preloadStatus, err := probePreloadStatus(ctx, database)
	if err == nil {
		result.AGEPreloadStatus = preloadStatus
	} else if !isInsufficientPrivilege(err) {
		return DegradedProbe{}, err
	}

	_, err = database.Exec(ctx, `SELECT ag_catalog.age_pi()`)
	switch {
	case err == nil:
		result.AGELoadabilityStatus = ProbePass
	case isInsufficientPrivilege(err):
		result.AGELoadabilityStatus = ProbeUnknown
		result.AGELoadabilityDetail = "permission denied while testing Apache AGE loadability"
	case isUndefinedAGEObject(err):
		result.AGELoadabilityStatus = ProbeFail
		result.AGELoadabilityDetail = "Apache AGE catalog function is unavailable"
	case isConnectionError(err):
		return DegradedProbe{}, fmt.Errorf("probe Apache AGE loadability: %w", err)
	default:
		result.AGELoadabilityStatus = ProbeFail
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) {
			result.AGELoadabilityDetail = fmt.Sprintf(
				"Apache AGE loadability probe failed (SQLSTATE %s)",
				pgErr.Code,
			)
		} else {
			result.AGELoadabilityDetail = "Apache AGE loadability probe failed"
		}
	}
	return result, nil
}

func isInsufficientPrivilege(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42501"
}

func isUndefinedAGEObject(err error) bool {
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return false
	}
	return pgErr.Code == "3F000" || pgErr.Code == "42883"
}

func isConnectionError(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && strings.HasPrefix(pgErr.Code, "08")
}
