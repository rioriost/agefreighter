package age

import (
	"errors"
	"testing"
)

func TestGraphIDRoundTrip(t *testing.T) {
	tests := []struct {
		label uint16
		entry uint64
	}{
		{label: 1, entry: 1},
		{label: 42, entry: 999},
		{label: MaxLabelID, entry: MaxEntryID},
	}
	for _, test := range tests {
		id, err := MakeGraphID(test.label, test.entry)
		if err != nil {
			t.Fatalf("MakeGraphID(%d, %d) error = %v", test.label, test.entry, err)
		}
		if id.LabelID() != test.label || id.EntryID() != test.entry {
			t.Errorf(
				"graphid = (%d, %d), want (%d, %d)",
				id.LabelID(),
				id.EntryID(),
				test.label,
				test.entry,
			)
		}
		if err := id.Validate(); err != nil {
			t.Errorf("Validate() error = %v", err)
		}
	}
}

func TestGraphIDRejectsInvalidParts(t *testing.T) {
	for _, input := range []struct {
		label uint16
		entry uint64
	}{
		{label: 0, entry: 1},
		{label: 1, entry: 0},
		{label: 1, entry: MaxEntryID + 1},
	} {
		if _, err := MakeGraphID(input.label, input.entry); !errors.Is(err, ErrInvalidGraphID) {
			t.Errorf("MakeGraphID(%d, %d) error = %v", input.label, input.entry, err)
		}
	}
	if err := GraphID(0).Validate(); !errors.Is(err, ErrInvalidGraphID) {
		t.Fatalf("GraphID(0).Validate() error = %v", err)
	}
}

func FuzzGraphIDRoundTrip(f *testing.F) {
	f.Add(uint16(1), uint64(1))
	f.Add(MaxLabelID, MaxEntryID)
	f.Fuzz(func(t *testing.T, label uint16, entry uint64) {
		id, err := MakeGraphID(label, entry)
		if label == 0 || entry == 0 || entry > MaxEntryID {
			if err == nil {
				t.Fatal("MakeGraphID() accepted invalid components")
			}
			return
		}
		if err != nil {
			t.Fatalf("MakeGraphID() error = %v", err)
		}
		if id.LabelID() != label || id.EntryID() != entry {
			t.Fatalf("round trip = (%d, %d)", id.LabelID(), id.EntryID())
		}
	})
}
