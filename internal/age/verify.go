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
	_, err := transaction.VerifyLabelRowsForIdentityCoverage(
		ctx, label, expectedRows, true,
	)
	return err
}

func (transaction *Transaction) VerifyLabelRowsForIdentityCoverage(
	ctx context.Context,
	label LabelCatalog,
	identityRows int64,
	requireFullCoverage bool,
) (int64, error) {
	if identityRows < 0 {
		return 0, errors.New("identity row count cannot be negative")
	}
	current, err := transaction.LookupLabel(ctx, label.GraphName, label.LabelName)
	if err != nil {
		return 0, err
	}
	if current != label {
		return 0, fmt.Errorf("label %q catalog changed before verification", label.LabelName)
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
		return 0, fmt.Errorf("verify label %s: %w", table, err)
	}
	if requireFullCoverage && actualRows != identityRows {
		return actualRows, fmt.Errorf(
			"label %s has %d rows, expected %d",
			table,
			actualRows,
			identityRows,
		)
	}
	if wrongLabelIDRows != 0 {
		return actualRows, fmt.Errorf(
			"label %s contains %d rows with the wrong graphid label",
			table,
			wrongLabelIDRows,
		)
	}
	return actualRows, nil
}
