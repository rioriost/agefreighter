package config

import "slices"

type StaticPlan struct {
	APIVersion string       `json:"apiVersion"`
	Job        string       `json:"job"`
	Source     PlanSource   `json:"source"`
	Target     PlanTarget   `json:"target"`
	Trial      *PlanTrial   `json:"trial,omitempty"`
	Limits     PlanLimits   `json:"limits"`
	Policies   PlanPolicies `json:"policies"`
	Warnings   []string     `json:"warnings,omitempty"`
}

type PlanTrial struct {
	MaxVerticesPerLabel int      `json:"maxVerticesPerLabel"`
	MaxVertices         int      `json:"maxVertices"`
	MaxEdges            int      `json:"maxEdges"`
	MaxBytes            string   `json:"maxBytes"`
	IncludeLabels       []string `json:"includeLabels,omitempty"`
}

type PlanSource struct {
	Type               SourceType            `json:"type"`
	Namespace          string                `json:"namespace"`
	PostgreSQLReadMode PostgreSQLReadMode    `json:"postgresqlReadMode,omitempty"`
	Neo4jMultiLabel    Neo4jMultiLabelPolicy `json:"neo4jMultiLabelPolicy,omitempty"`
	Neo4jDiscovery     bool                  `json:"neo4jDiscovery,omitempty"`
	CosmosGremlin      bool                  `json:"cosmosGremlin,omitempty"`
	FetchRows          int                   `json:"fetchRows,omitempty"`
	Consistency        string                `json:"consistency,omitempty"`
}

type PlanTarget struct {
	Type            TargetType            `json:"type"`
	Graph           string                `json:"graph"`
	Mode            LoadMode              `json:"mode"`
	PropertyMode    PropertyMode          `json:"propertyMode"`
	AppendDuplicate AppendDuplicatePolicy `json:"appendDuplicate,omitempty"`
}

type PlanLimits struct {
	MemoryLimit             string `json:"memoryLimit"`
	BatchRows               int    `json:"batchRows"`
	BatchBytes              string `json:"batchBytes"`
	MaxSourceConcurrency    int    `json:"maxSourceConcurrency"`
	MaxTransformConcurrency int    `json:"maxTransformConcurrency"`
	MaxTargetConnections    int    `json:"maxTargetConnections"`
	OperationTimeout        string `json:"operationTimeout"`
}

type PlanPolicies struct {
	MalformedRecord  MalformedRecordPolicy `json:"malformedRecord"`
	MissingEndpoint  MissingEndpointPolicy `json:"missingEndpoint"`
	RejectLimit      int                   `json:"rejectLimit"`
	MaxDeferredEdges int                   `json:"maxDeferredEdges,omitempty"`
}

func BuildStaticPlan(job LoadJob) StaticPlan {
	plan := StaticPlan{
		APIVersion: job.APIVersion,
		Job:        job.Metadata.Name,
		Source: PlanSource{
			Type:      job.Source.Type,
			Namespace: job.Source.Namespace,
		},
		Target: PlanTarget{
			Type:            job.Target.Type,
			Graph:           job.Target.Graph,
			Mode:            job.Target.Mode,
			PropertyMode:    job.Target.PropertyMode,
			AppendDuplicate: job.Target.AppendDuplicate,
		},
		Limits: PlanLimits{
			MemoryLimit:             job.Runtime.MemoryLimit.String(),
			BatchRows:               job.Runtime.BatchRows,
			BatchBytes:              job.Runtime.BatchBytes.String(),
			MaxSourceConcurrency:    job.Runtime.MaxSourceConcurrency,
			MaxTransformConcurrency: job.Runtime.MaxTransformConcurrency,
			MaxTargetConnections:    job.Runtime.MaxTargetConnections,
			OperationTimeout:        job.Runtime.OperationTimeout.String(),
		},
		Policies: PlanPolicies{
			MalformedRecord:  job.Errors.MalformedRecord,
			MissingEndpoint:  job.Errors.MissingEndpoint,
			RejectLimit:      job.Errors.RejectLimit,
			MaxDeferredEdges: job.Errors.MaxDeferredEdges,
		},
	}
	if job.Trial != nil && job.Trial.Enabled {
		plan.Trial = &PlanTrial{
			MaxVerticesPerLabel: job.Trial.MaxVerticesPerLabel,
			MaxVertices:         job.Trial.MaxVertices,
			MaxEdges:            job.Trial.MaxEdges,
			MaxBytes:            job.Trial.MaxBytes.String(),
			IncludeLabels:       slices.Clone(job.Trial.IncludeLabels),
		}
		plan.Warnings = append(plan.Warnings,
			"trial mode is deterministic, non-resumable, and emits only edges between selected vertices")
	}
	if job.Source.Type == SourcePostgreSQL && job.Source.PostgreSQL != nil {
		plan.Source.PostgreSQLReadMode = job.Source.PostgreSQL.ReadMode
		plan.Source.FetchRows = job.Source.PostgreSQL.FetchRows
		plan.Source.Consistency = "exported-repeatable-read-snapshot"
	}
	if job.Source.Type == SourceNeo4j && job.Source.Neo4j != nil {
		plan.Source.FetchRows = job.Source.Neo4j.FetchRows
		plan.Source.Neo4jMultiLabel = job.Source.Neo4j.MultiLabelPolicy
		plan.Source.Neo4jDiscovery = job.Source.Neo4j.Discovery != nil &&
			job.Source.Neo4j.Discovery.Enabled
		plan.Source.Consistency = "per-mapping-read-transaction"
		plan.Warnings = append(plan.Warnings,
			"Neo4j mappings are separate read transactions and do not observe one point-in-time graph snapshot")
		if plan.Source.Neo4jDiscovery {
			plan.Warnings = append(plan.Warnings,
				"Neo4j labels, relationship endpoint pairs, and properties are resolved before target admission")
		}
	}
	switch job.Source.Type {
	case SourceCosmos:
		if job.Source.Cosmos != nil {
			plan.Source.CosmosGremlin =
				job.Source.Cosmos.Gremlin != nil &&
					job.Source.Cosmos.Gremlin.Enabled
		}
		plan.Warnings = append(plan.Warnings,
			"source consistency capabilities are verified when the connector initializes")
		if plan.Source.CosmosGremlin {
			plan.Warnings = append(plan.Warnings,
				"Cosmos Gremlin labels and endpoint pairs are interpreted before target admission")
		}
	}
	if job.Target.Mode == LoadAppend || job.Target.Mode == LoadUpsert {
		plan.Warnings = append(plan.Warnings,
			"incremental modes require a compatible target identity catalog")
	}
	return plan
}
