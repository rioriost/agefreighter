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
	var graph meta.GraphGeneration
	labels := make([]age.LoadLabel, 0, len(kinds))
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if err := transaction.CreateGraph(ctx, job.Target.Graph); err != nil {
			return err
		}
		for name, kind := range kinds {
			if err := transaction.CreateLabel(ctx, job.Target.Graph, name, kind); err != nil {
				return err
			}
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		graphCatalog, err := transaction.LookupGraph(ctx, job.Target.Graph)
		if err != nil {
			return err
		}
		graph, err = transactionStore.RegisterGraphGeneration(ctx, meta.GraphGeneration{
			JobID: jobID, GraphName: graphCatalog.Name,
			GraphOID: graphCatalog.GraphOID, NamespaceOID: graphCatalog.NamespaceOID,
			Generation: 1, State: meta.GenerationLoading,
		})
		if err != nil {
			return err
		}
		for name, kind := range kinds {
			catalog, err := transaction.LookupLabel(ctx, job.Target.Graph, name)
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
	graphCatalog, err := adapter.LookupGraph(ctx, job.Target.Graph)
	if err != nil {
		return meta.GraphGeneration{}, nil, err
	}
	graph, err := store.AdmitGraphGeneration(ctx, storedJob.ID, meta.GraphGeneration{
		ID: storedJob.GraphGenerationID, JobID: storedJob.ID,
		GraphName: graphCatalog.Name, GraphOID: graphCatalog.GraphOID,
		NamespaceOID: graphCatalog.NamespaceOID, Generation: 1,
		State: meta.GenerationLoading,
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
		catalog, err := adapter.LookupLabel(ctx, job.Target.Graph, name)
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
