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
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		for name, kind := range kinds {
			if err := transaction.CreateLabel(ctx, graphName, name, kind); err != nil {
				return err
			}
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
		graphCatalog, err := transaction.LookupGraph(ctx, graphName)
		if err != nil {
			return err
		}
		graph, err = transactionStore.RegisterGraphGeneration(ctx, meta.GraphGeneration{
			JobID: jobID, GraphName: graphCatalog.Name,
			GraphOID: graphCatalog.GraphOID, NamespaceOID: graphCatalog.NamespaceOID,
			ReplacesGraphOID: replacesGraphOID,
			Generation:       generation, State: meta.GenerationLoading,
		})
		if err != nil {
			return err
		}
		for name, kind := range kinds {
			catalog, err := transaction.LookupLabel(ctx, graphName, name)
			if err != nil {
				return err
			}
			generation, err := transactionStore.RegisterLabelGeneration(ctx, meta.LabelGeneration{
				GraphGenerationID: graph.ID, LabelName: name,
				Kind: meta.LabelKind(kind), GraphNamespaceOID: catalog.NamespaceOID,
				LabelID: catalog.LabelID, RelationOID: catalog.RelationOID,
				SequenceOID: catalog.SequenceOID, MappingGeneration: 1,
			})
			if err != nil {
				return err
			}
			labels = append(labels, age.LoadLabel{Catalog: catalog, Generation: generation})
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
		ID: storedJob.GraphGenerationID, JobID: storedJob.ID,
		GraphName: graphCatalog.Name, GraphOID: graphCatalog.GraphOID,
		NamespaceOID:     graphCatalog.NamespaceOID,
		ReplacesGraphOID: storedGraph.ReplacesGraphOID,
		Generation:       storedGraph.Generation, State: meta.GenerationLoading,
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

func loadGraphName(job config.LoadJob, jobID string) (string, error) {
	switch job.Target.Mode {
	case config.LoadCreate:
		return job.Target.Graph, nil
	case config.LoadReplace:
		return age.DeriveGraphName(job.Target.Graph, age.ShadowName, jobID)
	default:
		return "", fmt.Errorf("load mode %q is not implemented", job.Target.Mode)
	}
}

func configuredLabels(job config.LoadJob) (map[string]age.LabelKind, error) {
	labels := make(map[string]age.LabelKind)
	for _, vertex := range job.Source.CSV.Vertices {
		labels[vertex.Label] = age.VertexLabel
	}
	for _, edge := range job.Source.CSV.Edges {
		if labels[edge.Label] == age.VertexLabel {
			return nil, fmt.Errorf("label %q is mapped as both vertex and edge", edge.Label)
		}
		if labels[edge.Start.Label] != age.VertexLabel {
			return nil, fmt.Errorf("edge label %q start label %q has no vertex mapping", edge.Label, edge.Start.Label)
		}
		if labels[edge.End.Label] != age.VertexLabel {
			return nil, fmt.Errorf("edge label %q end label %q has no vertex mapping", edge.Label, edge.End.Label)
		}
		labels[edge.Label] = age.EdgeLabel
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
