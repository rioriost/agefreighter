package app

import (
	"context"
	"errors"
	"fmt"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

func promoteReplace(
	ctx context.Context,
	adapter *age.Adapter,
	job config.LoadJob,
	jobID string,
	graph meta.GraphGeneration,
) error {
	backupGraph, err := age.DeriveGraphName(
		job.Target.Graph,
		age.BackupName,
		jobID,
	)
	if err != nil {
		return err
	}
	err = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if err := transaction.LockGraphLifecycle(ctx, job.Target.Graph); err != nil {
			return err
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		promotion, err := transactionStore.PrepareReplacePromotion(
			ctx,
			jobID,
			graph.ID,
		)
		if err != nil {
			return err
		}

		targetCatalog, err := transaction.LookupGraph(ctx, job.Target.Graph)
		if err != nil {
			return fmt.Errorf("lookup replacement target: %w", err)
		}
		if targetCatalog.GraphOID != promotion.NewGeneration.ReplacesGraphOID {
			return fmt.Errorf(
				"%w: replacement target OID changed from %d to %d",
				meta.ErrGenerationMismatch,
				promotion.NewGeneration.ReplacesGraphOID,
				targetCatalog.GraphOID,
			)
		}
		shadowCatalog, err := transaction.LookupGraph(
			ctx,
			promotion.NewGeneration.GraphName,
		)
		if err != nil {
			return fmt.Errorf("lookup replacement shadow: %w", err)
		}
		if shadowCatalog.GraphOID != promotion.NewGeneration.GraphOID {
			return fmt.Errorf(
				"%w: replacement shadow OID changed from %d to %d",
				meta.ErrGenerationMismatch,
				promotion.NewGeneration.GraphOID,
				shadowCatalog.GraphOID,
			)
		}
		if _, err := transaction.LookupGraph(ctx, backupGraph); err == nil {
			return fmt.Errorf("replacement backup graph %q already exists", backupGraph)
		} else if !errors.Is(err, age.ErrCatalogEntryNotFound) {
			return fmt.Errorf("check replacement backup graph: %w", err)
		}
		if err := transaction.PreflightGraphRename(ctx, targetCatalog); err != nil {
			return err
		}
		if err := transaction.PreflightGraphRename(ctx, shadowCatalog); err != nil {
			return err
		}
		if err := verifyGenerationTransaction(
			ctx,
			transaction,
			transactionStore,
			job,
			promotion.NewGeneration,
		); err != nil {
			return fmt.Errorf("verify replacement shadow: %w", err)
		}

		if err := transaction.RenameGraph(
			ctx,
			job.Target.Graph,
			backupGraph,
		); err != nil {
			return err
		}
		if err := transaction.RenameGraph(
			ctx,
			promotion.NewGeneration.GraphName,
			job.Target.Graph,
		); err != nil {
			return err
		}
		return transactionStore.CompleteReplacePromotion(
			ctx,
			promotion,
			job.Target.Graph,
			backupGraph,
		)
	})
	if err != nil {
		return err
	}
	adapter.ResetSessions()
	return nil
}

func Cleanup(ctx context.Context, path, jobID string) (meta.Job, error) {
	job, err := config.Load(path)
	if err != nil {
		return meta.Job{}, fmt.Errorf("load target configuration: %w", err)
	}
	if job.Target.Mode != config.LoadReplace {
		return meta.Job{}, errors.New("cleanup requires a replace load job")
	}
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		return meta.Job{}, err
	}
	defer adapter.Close()

	err = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if err := transaction.LockGraphLifecycle(ctx, job.Target.Graph); err != nil {
			return err
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		cleanup, err := transactionStore.PrepareBackupCleanup(ctx, jobID)
		if err != nil {
			return err
		}
		if cleanup.Job.TargetGraph != job.Target.Graph {
			return fmt.Errorf(
				"%w: cleanup target %q does not match job target %q",
				meta.ErrGenerationMismatch,
				job.Target.Graph,
				cleanup.Job.TargetGraph,
			)
		}
		if cleanup.AlreadyCleaned {
			return nil
		}
		targetCatalog, err := transaction.LookupGraph(ctx, cleanup.Job.TargetGraph)
		if err != nil {
			return fmt.Errorf("lookup active replacement graph: %w", err)
		}
		if targetCatalog.GraphOID != cleanup.Generation.GraphOID {
			return fmt.Errorf(
				"%w: active graph OID changed from %d to %d",
				meta.ErrGenerationMismatch,
				cleanup.Generation.GraphOID,
				targetCatalog.GraphOID,
			)
		}
		backupCatalog, err := transaction.LookupGraph(
			ctx,
			cleanup.Job.BackupGraphName,
		)
		if err != nil {
			return fmt.Errorf("lookup replacement backup: %w", err)
		}
		if backupCatalog.GraphOID != cleanup.Generation.ReplacesGraphOID {
			return fmt.Errorf(
				"%w: backup graph OID changed from %d to %d",
				meta.ErrGenerationMismatch,
				cleanup.Generation.ReplacesGraphOID,
				backupCatalog.GraphOID,
			)
		}
		if err := transaction.DropGraph(
			ctx,
			cleanup.Job.BackupGraphName,
			true,
		); err != nil {
			return err
		}
		return transactionStore.CompleteBackupCleanup(ctx, cleanup)
	})
	if err != nil {
		return meta.Job{}, err
	}
	adapter.ResetSessions()
	return store.GetJob(ctx, jobID)
}

func verifyGenerationTransaction(
	ctx context.Context,
	transaction *age.Transaction,
	store *meta.Store,
	job config.LoadJob,
	graph meta.GraphGeneration,
) error {
	graphCatalog, err := transaction.LookupGraph(ctx, graph.GraphName)
	if err != nil {
		return fmt.Errorf("verify graph catalog: %w", err)
	}
	current := graph
	current.GraphName = graphCatalog.Name
	current.GraphOID = graphCatalog.GraphOID
	current.NamespaceOID = graphCatalog.NamespaceOID
	admitted, err := store.AdmitGraphGeneration(ctx, graph.JobID, current)
	if err != nil {
		return fmt.Errorf("verify graph generation: %w", err)
	}
	kinds, err := configuredLabels(job)
	if err != nil {
		return err
	}
	for name, kind := range kinds {
		catalog, err := transaction.LookupLabel(ctx, graph.GraphName, name)
		if err != nil {
			return fmt.Errorf("verify label catalog %q: %w", name, err)
		}
		generation, err := store.AdmitLabelGeneration(ctx, admitted.ID, meta.LabelGeneration{
			ID: 1, GraphGenerationID: admitted.ID, LabelName: name,
			Kind: meta.LabelKind(kind), GraphNamespaceOID: catalog.NamespaceOID,
			LabelID: catalog.LabelID, RelationOID: catalog.RelationOID,
			SequenceOID: catalog.SequenceOID, MappingGeneration: 1,
		})
		if err != nil {
			return fmt.Errorf("verify label generation %q: %w", name, err)
		}
		expected, err := store.CountLabelIdentities(
			ctx,
			admitted.ID,
			generation.ID,
			generation.Kind,
		)
		if err != nil {
			return err
		}
		if err := transaction.VerifyLabelRows(ctx, catalog, expected); err != nil {
			return err
		}
	}
	return nil
}
