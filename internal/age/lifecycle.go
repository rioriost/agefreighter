package age

import (
	"context"
	"errors"
	"fmt"
)

const graphLifecycleLockSeed int64 = 0x6167656672

func (transaction *Transaction) LockGraphLifecycle(
	ctx context.Context,
	name string,
) error {
	if err := ValidateGraphName(name); err != nil {
		return err
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_xact_lock(
			pg_catalog.hashtextextended($1, $2)
		)`,
		"agefreighter:graph-lifecycle:"+name,
		graphLifecycleLockSeed,
	); err != nil {
		return fmt.Errorf("lock graph lifecycle %q: %w", name, err)
	}
	return nil
}

func (transaction *Transaction) TryLockGraphLifecycle(
	ctx context.Context,
	name string,
) (bool, error) {
	if err := ValidateGraphName(name); err != nil {
		return false, err
	}
	var locked bool
	if err := transaction.tx.QueryRow(
		ctx,
		`SELECT pg_catalog.pg_try_advisory_xact_lock(
			pg_catalog.hashtextextended($1, $2)
		)`,
		"agefreighter:graph-lifecycle:"+name,
		graphLifecycleLockSeed,
	).Scan(&locked); err != nil {
		return false, fmt.Errorf("try lock graph lifecycle %q: %w", name, err)
	}
	return locked, nil
}

func (transaction *Transaction) PreflightGraphRename(
	ctx context.Context,
	graph GraphCatalog,
) error {
	if err := ValidateGraphName(graph.Name); err != nil {
		return err
	}
	if graph.NamespaceOID == 0 || graph.GraphOID != graph.NamespaceOID {
		return errors.New("valid graph catalog is required for rename preflight")
	}
	var ownsSchema, canCreateSchema bool
	if err := transaction.tx.QueryRow(
		ctx,
		`SELECT
			pg_catalog.pg_has_role(current_user, nspowner, 'USAGE'),
			pg_catalog.has_database_privilege(
				current_user, current_database(), 'CREATE'
			)
		 FROM pg_catalog.pg_namespace
		 WHERE oid = $1`,
		graph.NamespaceOID,
	).Scan(&ownsSchema, &canCreateSchema); err != nil {
		return fmt.Errorf("preflight graph rename %q: %w", graph.Name, err)
	}
	if !ownsSchema {
		return fmt.Errorf(
			"preflight graph rename %q: current user does not own the graph schema",
			graph.Name,
		)
	}
	if !canCreateSchema {
		return fmt.Errorf(
			"preflight graph rename %q: current user lacks CREATE on the database",
			graph.Name,
		)
	}
	return nil
}

func (transaction *Transaction) CreateGraph(ctx context.Context, name string) error {
	if err := ValidateGraphName(name); err != nil {
		return err
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"SELECT ag_catalog.create_graph($1::name)",
		name,
	); err != nil {
		return fmt.Errorf("create graph %q: %w", name, err)
	}
	return nil
}

func (transaction *Transaction) DropGraph(
	ctx context.Context,
	name string,
	cascade bool,
) error {
	if err := ValidateGraphName(name); err != nil {
		return err
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"SELECT ag_catalog.drop_graph($1::name, $2)",
		name,
		cascade,
	); err != nil {
		return fmt.Errorf("drop graph %q: %w", name, err)
	}
	return nil
}

func (transaction *Transaction) RenameGraph(
	ctx context.Context,
	oldName string,
	newName string,
) error {
	if err := ValidateGraphName(oldName); err != nil {
		return err
	}
	if err := ValidateGraphName(newName); err != nil {
		return err
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`SELECT ag_catalog.alter_graph(
			$1::name,
			textout($2::text),
			$3::name
		)`,
		oldName,
		"RENAME",
		newName,
	); err != nil {
		return fmt.Errorf("rename graph %q to %q: %w", oldName, newName, err)
	}
	return nil
}

func (transaction *Transaction) CreateLabel(
	ctx context.Context,
	graphName string,
	labelName string,
	kind LabelKind,
) error {
	if err := ValidateGraphName(graphName); err != nil {
		return err
	}
	if err := ValidateLabelName(labelName); err != nil {
		return err
	}
	var function string
	switch kind {
	case VertexLabel:
		function = "create_vlabel"
	case EdgeLabel:
		function = "create_elabel"
	default:
		return fmt.Errorf("invalid label kind %d", kind)
	}
	query := fmt.Sprintf(
		"SELECT ag_catalog.%s(textout($1::text), textout($2::text))",
		function,
	)
	if _, err := transaction.tx.Exec(ctx, query, graphName, labelName); err != nil {
		return fmt.Errorf(
			"create %s label %q in graph %q: %w",
			kind,
			labelName,
			graphName,
			err,
		)
	}
	return nil
}

func (transaction *Transaction) DropLabel(
	ctx context.Context,
	graphName string,
	labelName string,
	force bool,
) error {
	if err := ValidateGraphName(graphName); err != nil {
		return err
	}
	if err := ValidateLabelName(labelName); err != nil {
		return err
	}
	if force {
		return errors.New("Apache AGE 1.6 does not support forced label drops")
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"SELECT ag_catalog.drop_label($1::name, $2::name, $3)",
		graphName,
		labelName,
		force,
	); err != nil {
		return fmt.Errorf(
			"drop label %q in graph %q: %w",
			labelName,
			graphName,
			err,
		)
	}
	return nil
}
