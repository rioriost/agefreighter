package app

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

func createCatalog(
	ctx context.Context,
	adapter *age.Adapter,
	job config.LoadJob,
	jobID string,
) (meta.GraphGeneration, []age.LoadLabel, error) {
	kinds, err := configuredLabels(job)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	graphName, err := loadGraphName(job, jobID)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	var graph meta.GraphGeneration
	labels := make([]age.LoadLabel, 0, len(kinds))
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		var replacesGraphOID uint32
		generation := uint64(1)
		if job.Target.Mode == config.LoadReplace {
			if err := transaction.LockGraphLifecycle(ctx, job.Target.Graph); err != nil {
				return err
			}
			target, err := transaction.LookupGraph(ctx, job.Target.Graph)
			if err != nil {
				return fmt.Errorf("lookup replacement target: %w", err)
			}
			replacesGraphOID = target.GraphOID
		}
		if err := transaction.CreateGraphWithLabels(ctx, graphName, kinds); err != nil {
			return err
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		if job.Target.Mode == config.LoadReplace {
			generation, err = transactionStore.NextGraphGeneration(ctx, job.Target.Graph)
			if err != nil {
				return err
			}
		}
		graph, err = transaction.RegisterCreatedGraph(
			ctx, jobID, graphName, replacesGraphOID, generation,
		)
		if err != nil {
			return err
		}
		for name, kind := range kinds {
			label, err := transaction.RegisterCreatedLabel(
				ctx, graph.ID, graphName, name, kind,
			)
			if err != nil {
				return err
			}
			labels = append(labels, label)
		}
		return nil
	}); err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	return graph, labels, nil
}

func admitCatalog(
	ctx context.Context,
	adapter *age.Adapter,
	store *meta.Store,
	job config.LoadJob,
	storedJob meta.Job,
) (meta.GraphGeneration, []age.LoadLabel, error) {
	storedGraph, err := store.GraphGenerationForJob(ctx, storedJob.ID)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	graphName, err := loadGraphName(job, storedJob.ID)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	graphCatalog, err := adapter.LookupGraph(ctx, graphName)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	if job.Target.Mode == config.LoadReplace {
		target, targetErr := adapter.LookupGraph(ctx, job.Target.Graph)
		if targetErr != nil {
			return meta.GraphGeneration{}, nil, fmt.Errorf(
				"lookup replacement target: %w",
				targetErr,
			)
		}
		if target.GraphOID != storedGraph.ReplacesGraphOID {
			return meta.GraphGeneration{}, nil, fmt.Errorf(
				"%w: replacement target OID changed from %d to %d",
				meta.ErrGenerationMismatch,
				storedGraph.ReplacesGraphOID,
				target.GraphOID,
			)
		}
	}
	graph, err := store.AdmitGraphGeneration(ctx, storedJob.ID, meta.GraphGeneration{
		ID: storedJob.GraphGenerationID, JobID: storedGraph.JobID,
		GraphName: graphCatalog.Name, GraphOID: graphCatalog.GraphOID,
		NamespaceOID:     graphCatalog.NamespaceOID,
		ReplacesGraphOID: storedGraph.ReplacesGraphOID,
		Generation:       storedGraph.Generation, State: storedGraph.State,
	})
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	kinds, err := configuredLabels(job)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	labels := make([]age.LoadLabel, 0, len(kinds))
	for name, kind := range kinds {
		catalog, err := adapter.LookupLabel(ctx, graphName, name)
		if err != nil {
			return meta.GraphGeneration{}, nil, err
		}
		generation, err := store.AdmitLabelGeneration(ctx, graph.ID, meta.LabelGeneration{
			ID: 1, GraphGenerationID: graph.ID, LabelName: name,
			Kind: meta.LabelKind(kind), GraphNamespaceOID: catalog.NamespaceOID,
			LabelID: catalog.LabelID, RelationOID: catalog.RelationOID,
			SequenceOID: catalog.SequenceOID, MappingGeneration: 1,
		})
		if err != nil {
			return meta.GraphGeneration{}, nil, err
		}
		labels = append(labels, age.LoadLabel{Catalog: catalog, Generation: generation})
	}
	return graph, labels, nil
}

func admitIncrementalCatalog(
	ctx context.Context,
	adapter *age.Adapter,
	job config.LoadJob,
	jobID string,
) (meta.GraphGeneration, []age.LoadLabel, error) {
	kinds, err := configuredLabels(job)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	var graph meta.GraphGeneration
	labels := make([]age.LoadLabel, 0, len(kinds))
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		locked, err := transaction.TryLockGraphLifecycle(
			ctx,
			job.Target.Graph,
		)
		if err != nil {
			return err
		}
		if !locked {
			return meta.ErrIncrementalConflict
		}
		graphCatalog, err := transaction.LookupGraph(ctx, job.Target.Graph)
		if err != nil {
			return err
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		graph, err = transactionStore.BindActiveGraphGeneration(
			ctx,
			jobID,
			job.Target.Graph,
		)
		if err != nil {
			return err
		}
		if graph.GraphOID != graphCatalog.GraphOID ||
			graph.NamespaceOID != graphCatalog.NamespaceOID {
			return fmt.Errorf(
				"%w: active graph %q catalog identity changed",
				meta.ErrGenerationMismatch,
				job.Target.Graph,
			)
		}
		for name, kind := range kinds {
			catalog, err := transaction.LookupLabel(ctx, job.Target.Graph, name)
			if err != nil {
				return err
			}
			generation, err := transactionStore.AdmitLabelGeneration(
				ctx,
				graph.ID,
				meta.LabelGeneration{
					ID: 1, GraphGenerationID: graph.ID, LabelName: name,
					Kind:              meta.LabelKind(kind),
					GraphNamespaceOID: catalog.NamespaceOID,
					LabelID:           catalog.LabelID,
					RelationOID:       catalog.RelationOID,
					SequenceOID:       catalog.SequenceOID,
					MappingGeneration: 1,
				},
			)
			if err != nil {
				return err
			}
			labels = append(
				labels,
				age.LoadLabel{Catalog: catalog, Generation: generation},
			)
		}
		return nil
	}); err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	return graph, labels, nil
}

func loadGraphName(job config.LoadJob, jobID string) (string, error) {
	switch job.Target.Mode {
	case config.LoadCreate, config.LoadAppend, config.LoadUpsert:
		return job.Target.Graph, nil
	case config.LoadReplace:
		return age.DeriveGraphName(job.Target.Graph, age.ShadowName, jobID)
	default:
		return "", fmt.Errorf("load mode %q is not implemented", job.Target.Mode)
	}
}

func configuredLabels(job config.LoadJob) (map[string]age.LabelKind, error) {
	labels := make(map[string]age.LabelKind)
	type configuredEdge struct {
		label string
		start string
		end   string
	}
	var vertices []string
	var edges []configuredEdge
	switch job.Source.Type {
	case config.SourceCSV:
		if job.Source.CSV == nil {
			return nil, errors.New("CSV source configuration is required")
		}
		for _, vertex := range job.Source.CSV.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.CSV.Edges {
			edges = append(edges, configuredEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	case config.SourceCosmos:
		if job.Source.Cosmos == nil {
			return nil, errors.New("Cosmos source configuration is required")
		}
		for _, vertex := range job.Source.Cosmos.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.Cosmos.Edges {
			edges = append(edges, configuredEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	case config.SourcePostgreSQL:
		if job.Source.PostgreSQL == nil {
			return nil, errors.New("PostgreSQL source configuration is required")
		}
		for _, vertex := range job.Source.PostgreSQL.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.PostgreSQL.Edges {
			edges = append(edges, configuredEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	case config.SourceNeo4j:
		if job.Source.Neo4j == nil {
			return nil, errors.New("Neo4j source configuration is required")
		}
		for _, vertex := range job.Source.Neo4j.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.Neo4j.Edges {
			edges = append(edges, configuredEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	default:
		return nil, fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
	for _, vertex := range vertices {
		labels[vertex] = age.VertexLabel
	}
	for _, edge := range edges {
		if labels[edge.label] == age.VertexLabel {
			return nil, fmt.Errorf("label %q is mapped as both vertex and edge", edge.label)
		}
		if labels[edge.start] != age.VertexLabel {
			return nil, fmt.Errorf("edge label %q start label %q has no vertex mapping", edge.label, edge.start)
		}
		if labels[edge.end] != age.VertexLabel {
			return nil, fmt.Errorf("edge label %q end label %q has no vertex mapping", edge.label, edge.end)
		}
		labels[edge.label] = age.EdgeLabel
	}
	return labels, nil
}

func failJob(
	ctx context.Context,
	store *meta.Store,
	jobID string,
	cause error,
) error {
	failureCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()
	return errors.Join(cause, store.FailJob(
		failureCtx,
		jobID,
		cause.Error(),
	))
}
