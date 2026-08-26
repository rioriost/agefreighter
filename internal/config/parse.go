package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"go.yaml.in/yaml/v3"
)

const MaxDocumentBytes = 4 << 20

// defaultCosmosPageSize is applied when a Cosmos source does not set an
// explicit pageSize; it is a conservative page-size hint that keeps
// individual pages small enough for bounded-memory iteration.
const defaultCosmosPageSize = 100

func Load(path string) (LoadJob, error) {
	file, err := os.Open(path)
	if err != nil {
		return LoadJob{}, fmt.Errorf("open configuration: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(io.LimitReader(file, MaxDocumentBytes+1))
	if err != nil {
		return LoadJob{}, fmt.Errorf("read configuration: %w", err)
	}
	if len(data) > MaxDocumentBytes {
		return LoadJob{}, fmt.Errorf("configuration exceeds %d bytes", MaxDocumentBytes)
	}
	job, err := Parse(data)
	if err != nil {
		return LoadJob{}, err
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return LoadJob{}, fmt.Errorf("resolve configuration directory: %w", err)
	}
	resolveJobPaths(&job, base)
	return job, nil
}

func resolveJobPaths(job *LoadJob, base string) {
	resolve := func(path string) string {
		if path == "" || filepath.IsAbs(path) {
			return path
		}
		return filepath.Join(base, path)
	}
	if job.Source.CSV != nil {
		for index := range job.Source.CSV.Vertices {
			job.Source.CSV.Vertices[index].Path = resolve(job.Source.CSV.Vertices[index].Path)
		}
		for index := range job.Source.CSV.Edges {
			job.Source.CSV.Edges[index].Path = resolve(job.Source.CSV.Edges[index].Path)
		}
	}
	if job.Source.PostgreSQL != nil {
		job.Source.PostgreSQL.Connection.File = resolve(
			job.Source.PostgreSQL.Connection.File,
		)
	}
	if job.Source.Neo4j != nil && job.Source.Neo4j.Password != nil {
		job.Source.Neo4j.Password.File = resolve(job.Source.Neo4j.Password.File)
	}
	job.Target.Connection.File = resolve(job.Target.Connection.File)
	job.Errors.QuarantinePath = resolve(job.Errors.QuarantinePath)
}

func Parse(data []byte) (LoadJob, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)

	var job LoadJob
	if err := decoder.Decode(&job); err != nil {
		return LoadJob{}, fmt.Errorf("decode configuration: %w", err)
	}

	var extra any
	err := decoder.Decode(&extra)
	if !errors.Is(err, io.EOF) {
		if err == nil {
			return LoadJob{}, errors.New("configuration must contain exactly one document")
		}
		return LoadJob{}, fmt.Errorf("decode trailing configuration: %w", err)
	}

	job.applyDefaults()
	if err := job.Validate(); err != nil {
		return LoadJob{}, err
	}
	return job, nil
}

func (job *LoadJob) applyDefaults() {
	if job.Source.PostgreSQL != nil {
		if job.Source.PostgreSQL.ReadMode == "" {
			job.Source.PostgreSQL.ReadMode = PostgreSQLReadCopy
		}
		if job.Source.PostgreSQL.FetchRows == 0 {
			job.Source.PostgreSQL.FetchRows = 1_000
		}
	}
	if job.Source.Neo4j != nil {
		if job.Source.Neo4j.FetchRows == 0 {
			job.Source.Neo4j.FetchRows = 1_000
		}
		if job.Source.Neo4j.MultiLabelPolicy == "" {
			job.Source.Neo4j.MultiLabelPolicy = Neo4jMultiLabelConfigured
		}
	}
	if job.Target.Mode == "" {
		job.Target.Mode = LoadCreate
	}
	if job.Target.PropertyMode == "" {
		job.Target.PropertyMode = PropertiesReplace
	}
	if job.Target.Mode == LoadAppend && job.Target.AppendDuplicate == "" {
		job.Target.AppendDuplicate = AppendDuplicateError
	}
	if job.Runtime.MemoryLimit == 0 {
		job.Runtime.MemoryLimit = 1 * gibibyte
	}
	if job.Runtime.BatchRows == 0 {
		job.Runtime.BatchRows = 10_000
	}
	if job.Runtime.BatchBytes == 0 {
		job.Runtime.BatchBytes = 16 * mebibyte
	}
	if job.Runtime.MaxSourceConcurrency == 0 {
		job.Runtime.MaxSourceConcurrency = 4
	}
	if job.Runtime.MaxTransformConcurrency == 0 {
		job.Runtime.MaxTransformConcurrency = 4
	}
	if job.Runtime.MaxTargetConnections == 0 {
		job.Runtime.MaxTargetConnections = 4
	}
	if job.Runtime.OperationTimeout == 0 {
		job.Runtime.OperationTimeout = Duration(30 * time.Second)
	}
	if job.Errors.MalformedRecord == "" {
		job.Errors.MalformedRecord = MalformedFail
	}
	if job.Errors.MissingEndpoint == "" {
		job.Errors.MissingEndpoint = MissingEndpointError
	}
	if (job.Target.Mode == LoadUpsert ||
		job.Errors.MissingEndpoint == MissingEndpointDefer) &&
		job.Errors.MaxDeferredEdges == 0 {
		job.Errors.MaxDeferredEdges = 100_000
	}
	if job.Source.CSV != nil {
		applyDelimitedDefaults(&job.Source.CSV.Defaults)
		for index := range job.Source.CSV.Vertices {
			if job.Source.CSV.Vertices[index].Format != nil {
				inheritDelimitedOptions(job.Source.CSV.Defaults, job.Source.CSV.Vertices[index].Format)
			}
		}
		for index := range job.Source.CSV.Edges {
			if job.Source.CSV.Edges[index].Format != nil {
				inheritDelimitedOptions(job.Source.CSV.Defaults, job.Source.CSV.Edges[index].Format)
			}
		}
	}
	if job.Source.Cosmos != nil && job.Source.Cosmos.Credential == "" {
		job.Source.Cosmos.Credential = "default-azure"
	}
	if job.Source.Cosmos != nil && job.Source.Cosmos.PageSize == 0 {
		job.Source.Cosmos.PageSize = defaultCosmosPageSize
	}
}

func applyDelimitedDefaults(options *DelimitedOptions) {
	if options.Delimiter == "" {
		options.Delimiter = ","
	}
	if options.Quote == "" {
		options.Quote = `"`
	}
	if options.Escape == "" {
		options.Escape = `"`
	}
	if options.Header == nil {
		header := true
		options.Header = &header
	}
	if options.Encoding == "" {
		options.Encoding = "utf-8"
	}
	if options.NullValue == nil {
		nullValue := ""
		options.NullValue = &nullValue
	}
}

func inheritDelimitedOptions(defaults DelimitedOptions, options *DelimitedOptions) {
	if options.Delimiter == "" {
		options.Delimiter = defaults.Delimiter
	}
	if options.Quote == "" {
		options.Quote = defaults.Quote
	}
	if options.Escape == "" {
		options.Escape = defaults.Escape
	}
	if options.Header == nil {
		header := *defaults.Header
		options.Header = &header
	}
	if options.Encoding == "" {
		options.Encoding = defaults.Encoding
	}
	if options.NullValue == nil {
		nullValue := *defaults.NullValue
		options.NullValue = &nullValue
	}
}
