package config

type StaticPlan struct {
	APIVersion string       `json:"apiVersion"`
	Job        string       `json:"job"`
	Source     PlanSource   `json:"source"`
	Target     PlanTarget   `json:"target"`
	Limits     PlanLimits   `json:"limits"`
	Policies   PlanPolicies `json:"policies"`
	Warnings   []string     `json:"warnings,omitempty"`
}

type PlanSource struct {
	Type      SourceType `json:"type"`
	Namespace string     `json:"namespace"`
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
	switch job.Source.Type {
	case SourceNeo4j, SourceCosmos:
		plan.Warnings = append(plan.Warnings,
			"source consistency capabilities are verified when the connector initializes")
	}
	if job.Target.Mode == LoadAppend || job.Target.Mode == LoadUpsert {
		plan.Warnings = append(plan.Warnings,
			"incremental modes require a compatible target identity catalog")
	}
	return plan
}
