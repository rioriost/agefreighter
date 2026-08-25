package age

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (transaction *Transaction) AnalyzeLabel(
	ctx context.Context,
	label LabelCatalog,
) error {
	current, err := transaction.LookupLabel(ctx, label.GraphName, label.LabelName)
	if err != nil {
		return err
	}
	if current != label {
		return fmt.Errorf("label %q catalog changed before ANALYZE", label.LabelName)
	}
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	if _, err := transaction.tx.Exec(ctx, "ANALYZE "+table); err != nil {
		return fmt.Errorf("analyze label %s: %w", table, err)
	}
	return nil
}

func (transaction *Transaction) VerifyLabelRows(
	ctx context.Context,
	label LabelCatalog,
	expectedRows int64,
) error {
	if expectedRows < 0 {
		return errors.New("expected row count cannot be negative")
	}
	current, err := transaction.LookupLabel(ctx, label.GraphName, label.LabelName)
	if err != nil {
		return err
	}
	if current != label {
		return fmt.Errorf("label %q catalog changed before verification", label.LabelName)
	}
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	var (
		actualRows       int64
		wrongLabelIDRows int64
	)
	if err := transaction.tx.QueryRow(
		ctx,
		fmt.Sprintf(
			`SELECT
				count(*),
				count(*) FILTER (
					WHERE ((id::text::bigint >> 48) & 65535) <> $1
				)
			 FROM %s`,
			table,
		),
		int32(label.LabelID),
	).Scan(&actualRows, &wrongLabelIDRows); err != nil {
		return fmt.Errorf("verify label %s: %w", table, err)
	}
	if actualRows != expectedRows {
		return fmt.Errorf(
			"label %s has %d rows, expected %d",
			table,
			actualRows,
			expectedRows,
		)
	}
	if wrongLabelIDRows != 0 {
		return fmt.Errorf(
			"label %s contains %d rows with the wrong graphid label",
			table,
			wrongLabelIDRows,
		)
	}
	return nil
}
