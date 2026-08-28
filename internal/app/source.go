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
	sourceneo4j "github.com/rioriost/agefreighter/internal/source/neo4j"
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
	case config.SourceNeo4j:
		if job.Source.Neo4j == nil {
			return errors.New("Neo4j source configuration is required")
		}
	default:
		return fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
	return nil
}

func resolveSource(
	ctx context.Context,
	job config.LoadJob,
) (config.LoadJob, error) {
	return resolveSourceBounded(ctx, job, nil)
}

func resolveSourceBounded(
	ctx context.Context,
	job config.LoadJob,
	budget *sourcecontract.ProfileBudget,
) (config.LoadJob, error) {
	switch job.Source.Type {
	case config.SourceNeo4j:
		return resolveNeo4jDiscovery(ctx, job, budget)
	case config.SourceCosmos:
		return resolveCosmosGremlin(ctx, job, budget)
	default:
		return job, nil
	}
}

func resolveNeo4jDiscovery(
	ctx context.Context,
	job config.LoadJob,
	budget *sourcecontract.ProfileBudget,
) (config.LoadJob, error) {
	source := job.Source.Neo4j
	if source == nil ||
		source.Discovery == nil ||
		!source.Discovery.Enabled {
		return job, nil
	}
	var password string
	var err error
	if source.Password != nil {
		password, err = resolveSecret(*source.Password)
		if err != nil {
			return config.LoadJob{}, fmt.Errorf(
				"resolve Neo4j source password for discovery: %w",
				err,
			)
		}
	}
	client, err := sourceneo4j.NewSDKClient(
		ctx,
		source.URI,
		source.Database,
		source.Username,
		password,
		source.FetchRows,
	)
	if err != nil {
		return config.LoadJob{}, err
	}
	resolved, discoverErr := sourceneo4j.DiscoverMappingsBounded(
		ctx,
		*source,
		client,
		budget,
	)
	closeErr := client.Close()
	if err := errors.Join(discoverErr, closeErr); err != nil {
		return config.LoadJob{}, fmt.Errorf(
			"discover Neo4j graph: %w",
			err,
		)
	}
	job.Source.Neo4j = &resolved
	if err := job.Validate(); err != nil {
		return config.LoadJob{}, fmt.Errorf(
			"validate discovered Neo4j mappings: %w",
			err,
		)
	}
	return job, nil
}

func resolveCosmosGremlin(
	ctx context.Context,
	job config.LoadJob,
	budget *sourcecontract.ProfileBudget,
) (config.LoadJob, error) {
	source := job.Source.Cosmos
	if source == nil ||
		source.Gremlin == nil ||
		!source.Gremlin.Enabled {
		return job, nil
	}
	client, err := sourcecosmos.NewSDKQueryClient(
		ctx,
		source.Endpoint,
		source.Database,
	)
	if err != nil {
		return config.LoadJob{}, err
	}
	resolved, interpretErr := sourcecosmos.InterpretGremlinDocumentsBounded(
		ctx,
		*source,
		client,
		budget,
	)
	closeErr := client.Close()
	if err := errors.Join(interpretErr, closeErr); err != nil {
		return config.LoadJob{}, fmt.Errorf(
			"interpret Cosmos Gremlin documents: %w",
			err,
		)
	}
	job.Source.Cosmos = &resolved
	if err := job.Validate(); err != nil {
		return config.LoadJob{}, fmt.Errorf(
			"validate interpreted Cosmos Gremlin mappings: %w",
			err,
		)
	}
	return job, nil
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
	case config.SourceNeo4j:
		var password string
		var err error
		if job.Source.Neo4j.Password != nil {
			password, err = resolveSecret(*job.Source.Neo4j.Password)
			if err != nil {
				return nil, fmt.Errorf("resolve Neo4j source password: %w", err)
			}
		}
		client, err := sourceneo4j.NewSDKClient(
			ctx,
			job.Source.Neo4j.URI,
			job.Source.Neo4j.Database,
			job.Source.Neo4j.Username,
			password,
			job.Source.Neo4j.FetchRows,
		)
		if err != nil {
			return nil, err
		}
		options := sourceneo4j.IteratorOptions{
			Namespace:           job.Source.Namespace,
			Source:              *job.Source.Neo4j,
			Client:              client,
			AfterToken:          afterToken,
			RejectLimit:         job.Errors.RejectLimit,
			PreencodeProperties: true,
		}
		if job.Errors.MalformedRecord == config.MalformedQuarantine {
			options.OnMalformed = func(
				ctx context.Context,
				malformed sourceneo4j.MalformedRecord,
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
		iterator, err := sourceneo4j.NewIterator(ctx, options)
		if err != nil {
			return nil, errors.Join(err, client.Close())
		}
		return iterator, nil
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
