package age

import (
	"errors"
	"fmt"
)

const (
	MaxLabelID uint16 = ^uint16(0)
	MaxEntryID uint64 = 1<<48 - 1
)

var ErrInvalidGraphID = errors.New("invalid graphid")

type GraphID int64

func MakeGraphID(labelID uint16, entryID uint64) (GraphID, error) {
	if labelID == 0 {
		return 0, fmt.Errorf("%w: label ID must be positive", ErrInvalidGraphID)
	}
	if entryID == 0 || entryID > MaxEntryID {
		return 0, fmt.Errorf(
			"%w: entry ID %d is outside 1..%d",
			ErrInvalidGraphID,
			entryID,
			MaxEntryID,
		)
	}
	bits := uint64(labelID)<<48 | entryID
	return GraphID(int64(bits)), nil
}

func (id GraphID) LabelID() uint16 {
	return uint16(uint64(id) >> 48)
}

func (id GraphID) EntryID() uint64 {
	return uint64(id) & MaxEntryID
}

func (id GraphID) Validate() error {
	if id.LabelID() == 0 || id.EntryID() == 0 {
		return fmt.Errorf(
			"%w: label ID %d, entry ID %d",
			ErrInvalidGraphID,
			id.LabelID(),
			id.EntryID(),
		)
	}
	return nil
}
