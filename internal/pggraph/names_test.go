package pggraph

import (
	"strings"
	"testing"
)

func TestQuoteIdentifier(t *testing.T) {
	if got := QuoteIdentifier(`Graph "Data"`); got != `"Graph ""Data"""` {
		t.Fatalf("QuoteIdentifier() = %q", got)
	}
}

func TestPhysicalNameIsStableBoundedAndCollisionResistant(t *testing.T) {
	first := PhysicalName("af_v_", strings.Repeat("Long Label ", 20))
	if len(first) > maxIdentifierBytes {
		t.Fatalf("PhysicalName() length = %d, want <= %d", len(first), maxIdentifierBytes)
	}
	if again := PhysicalName("af_v_", strings.Repeat("Long Label ", 20)); again != first {
		t.Fatalf("PhysicalName() is not stable: %q != %q", first, again)
	}
	if other := PhysicalName("af_v_", strings.Repeat("Long Label ", 19)+"Other"); other == first {
		t.Fatalf("PhysicalName() collision = %q", first)
	}
	if unicodeName := PhysicalName("af_v_", "供給先"); !strings.HasPrefix(unicodeName, "af_v_label_") {
		t.Fatalf("PhysicalName(unicode) = %q", unicodeName)
	}
}

func FuzzPhysicalName(f *testing.F) {
	f.Add("Person")
	f.Add("供給先 / Supplier")
	f.Add(strings.Repeat("x", 200))
	f.Fuzz(func(t *testing.T, label string) {
		name := PhysicalName("af_v_", label)
		if name == "" || len(name) > maxIdentifierBytes {
			t.Fatalf("PhysicalName(%q) = %q", label, name)
		}
		if again := PhysicalName("af_v_", label); again != name {
			t.Fatalf("PhysicalName(%q) changed: %q != %q", label, name, again)
		}
	})
}
