package age

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/jackc/pgx/v5"
)

type IDBlock struct {
	LabelID    uint16
	FirstEntry uint64
	LastEntry  uint64
}

func (block IDBlock) Count() uint64 {
	if block.FirstEntry == 0 || block.LastEntry < block.FirstEntry {
		return 0
	}
	return block.LastEntry - block.FirstEntry + 1
}

func (block IDBlock) GraphID(offset uint64) (GraphID, error) {
	if offset >= block.Count() {
		return 0, fmt.Errorf("%w: block offset %d", ErrInvalidGraphID, offset)
	}
	return MakeGraphID(block.LabelID, block.FirstEntry+offset)
}

func (transaction *Transaction) LockLabel(
	ctx context.Context,
	graphOID uint32,
	labelID uint16,
) error {
	if graphOID == 0 || labelID == 0 {
		return errors.New("graph OID and label ID must be positive")
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"SELECT pg_advisory_xact_lock($1::integer, $2::integer)",
		int32(graphOID),
		int32(labelID),
	); err != nil {
		return fmt.Errorf(
			"lock graph OID %d label ID %d: %w",
			graphOID,
			labelID,
			err,
		)
	}
	return nil
}

func (transaction *Transaction) ReserveIDs(
	ctx context.Context,
	expected LabelCatalog,
	count uint64,
) (IDBlock, error) {
	if count == 0 || count > MaxEntryID {
		return IDBlock{}, fmt.Errorf("reservation count must be within 1..%d", MaxEntryID)
	}
	if err := transaction.LockLabel(ctx, expected.GraphOID, expected.LabelID); err != nil {
		return IDBlock{}, err
	}
	current, err := transaction.LookupLabel(
		ctx,
		expected.GraphName,
		expected.LabelName,
	)
	if err != nil {
		return IDBlock{}, err
	}
	if current != expected {
		return IDBlock{}, fmt.Errorf(
			"label %q catalog changed before ID reservation",
			expected.LabelName,
		)
	}

	qualifiedSequence := pgx.Identifier{
		current.GraphName,
		current.SequenceName,
	}.Sanitize()
	var first int64
	if err := transaction.tx.QueryRow(
		ctx,
		"SELECT nextval($1::regclass)",
		qualifiedSequence,
	).Scan(&first); err != nil {
		return IDBlock{}, fmt.Errorf(
			"reserve first ID from sequence %s: %w",
			qualifiedSequence,
			err,
		)
	}
	if first <= 0 {
		return IDBlock{}, fmt.Errorf("sequence %s returned invalid ID %d", qualifiedSequence, first)
	}
	firstEntry := uint64(first)
	if count-1 > math.MaxUint64-firstEntry {
		return IDBlock{}, errors.New("ID reservation overflow")
	}
	lastEntry := firstEntry + count - 1
	if lastEntry > MaxEntryID {
		return IDBlock{}, fmt.Errorf(
			"ID reservation %d..%d exceeds AGE entry ID maximum %d",
			firstEntry,
			lastEntry,
			MaxEntryID,
		)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"SELECT setval($1::regclass, $2::bigint, true)",
		qualifiedSequence,
		int64(lastEntry),
	); err != nil {
		return IDBlock{}, fmt.Errorf(
			"advance sequence %s to %d: %w",
			qualifiedSequence,
			lastEntry,
			err,
		)
	}
	return IDBlock{
		LabelID:    current.LabelID,
		FirstEntry: firstEntry,
		LastEntry:  lastEntry,
	}, nil
}
