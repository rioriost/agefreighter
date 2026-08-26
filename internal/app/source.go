package app

import (
	"context"
	"errors"
	"fmt"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/reject"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	sourcecosmos "github.com/rioriost/agefreighter/internal/source/cosmos"
	sourcecsv "github.com/rioriost/agefreighter/internal/source/csv"
	sourcepostgres "github.com/rioriost/agefreighter/internal/source/postgres"
	"github.com/rioriost/agefreighter/pkg/model"
)

func validateImplementedSource(job config.LoadJob) error {
	switch job.Source.Type {
	case config.SourceCSV:
		if job.Source.CSV == nil {
			return errors.New("CSV source configuration is required")
		}
	case config.SourceCosmos:
		if job.Source.Cosmos == nil {
			return errors.New("Cosmos source configuration is required")
		}
	case config.SourcePostgreSQL:
		if job.Source.PostgreSQL == nil {
			return errors.New("PostgreSQL source configuration is required")
		}
	default:
		return fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
	return nil
}

func newSourceIterator(
	ctx context.Context,
	job config.LoadJob,
	afterToken string,
	quarantine *reject.JSONLWriter,
) (sourcecontract.Iterator, error) {
	switch job.Source.Type {
	case config.SourceCSV:
		options := sourcecsv.IteratorOptions{
			Namespace: job.Source.Namespace,
			Source:    *job.Source.CSV, AfterToken: afterToken,
			RejectLimit: job.Errors.RejectLimit, PreencodeProperties: true,
			OptimizeRFC4180: true,
		}
		if job.Errors.MalformedRecord == config.MalformedQuarantine {
			options.OnMalformed = func(ctx context.Context, malformed sourcecsv.MalformedRecord) error {
				return writeSourceRejection(
					ctx,
					quarantine,
					malformed.Position,
					malformed.Fields,
					malformed.Err,
				)
			}
		}
		return sourcecsv.NewIterator(ctx, options)
	case config.SourceCosmos:
		client, err := sourcecosmos.NewSDKQueryClient(
			ctx,
			job.Source.Cosmos.Endpoint,
			job.Source.Cosmos.Database,
		)
		if err != nil {
			return nil, err
		}
		options := sourcecosmos.IteratorOptions{
			Namespace: job.Source.Namespace,
			Source:    *job.Source.Cosmos, Client: client, AfterToken: afterToken,
			RejectLimit: job.Errors.RejectLimit, PreencodeProperties: true,
		}
		if job.Errors.MalformedRecord == config.MalformedQuarantine {
			options.OnMalformed = func(ctx context.Context, malformed sourcecosmos.MalformedRecord) error {
				return writeSourceRejection(ctx, quarantine, malformed.Position, nil, malformed.Err)
			}
		}
		iterator, err := sourcecosmos.NewIterator(ctx, options)
		if err != nil {
			return nil, errors.Join(err, client.Close())
		}
		return iterator, nil
	case config.SourcePostgreSQL:
		dsn, err := resolveSecret(job.Source.PostgreSQL.Connection)
		if err != nil {
			return nil, fmt.Errorf("resolve PostgreSQL source connection: %w", err)
		}
		options := sourcepostgres.IteratorOptions{
			Namespace:           job.Source.Namespace,
			Source:              *job.Source.PostgreSQL,
			DSN:                 dsn,
			AfterToken:          afterToken,
			RejectLimit:         job.Errors.RejectLimit,
			PreencodeProperties: true,
			MaxReaders:          job.Runtime.MaxSourceConcurrency,
		}
		if job.Errors.MalformedRecord == config.MalformedQuarantine {
			options.OnMalformed = func(
				ctx context.Context,
				malformed sourcepostgres.MalformedRecord,
			) error {
				return writeSourceRejection(
					ctx,
					quarantine,
					malformed.Position,
					nil,
					malformed.Err,
				)
			}
		}
		return sourcepostgres.NewIterator(ctx, options)
	default:
		return nil, fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
}

func writeSourceRejection(
	ctx context.Context,
	quarantine *reject.JSONLWriter,
	position model.SourcePosition,
	fields []string,
	cause error,
) error {
	if quarantine == nil {
		return errors.New("source quarantine writer is not configured")
	}
	return quarantine.Write(ctx, reject.Rejection{
		Fields: fields, Position: position,
		Code: "malformed-record", Message: cause.Error(),
	})
}

func sourceRejectionCheckpoint(
	iterator sourcecontract.Iterator,
) (int64, model.SourcePosition) {
	checkpointer, ok := iterator.(sourcecontract.RejectionCheckpointer)
	if !ok {
		return 0, model.SourcePosition{}
	}
	return checkpointer.RejectionCheckpoint()
}

func sourceTelemetry(iterator sourcecontract.Iterator) *sourcecontract.Telemetry {
	provider, ok := iterator.(sourcecontract.TelemetryProvider)
	if !ok {
		return nil
	}
	telemetry := provider.Telemetry()
	return &telemetry
}
