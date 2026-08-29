package source

import (
	"errors"
	"testing"
)

func TestProfileBudgetIsCumulativeAcrossCharges(t *testing.T) {
	budget := NewProfileBudget(ProfileBudgetLimits{
		Rows: 2, Pages: 2, RawInputBytes: 10, DecodedInputBytes: 10,
		RequestCharge: 3, Labels: 2, Properties: 2,
	})
	if err := budget.Charge(ProfileBudgetUsage{
		Rows: 1, Pages: 1, RawInputBytes: 5, DecodedInputBytes: 5,
		RequestCharge: 1, Labels: 1, Properties: 1,
	}); err != nil {
		t.Fatal(err)
	}
	if err := budget.Charge(ProfileBudgetUsage{
		Rows: 1, Pages: 1, RawInputBytes: 5, DecodedInputBytes: 5,
		RequestCharge: 2, Labels: 1, Properties: 1,
	}); err != nil {
		t.Fatal(err)
	}
	if err := budget.Full(); !errors.Is(err, ErrProfileBudget) {
		t.Fatalf("Full() error = %v", err)
	}
	usage, limit := budget.Snapshot()
	if usage.Rows != 2 || usage.Pages != 2 || usage.RawInputBytes != 10 ||
		usage.DecodedInputBytes != 10 || usage.RequestCharge != 3 ||
		usage.Labels != 2 || usage.Properties != 2 || limit == "" {
		t.Fatalf("snapshot = %#v, limit %q", usage, limit)
	}
}

func TestProfileBudgetFullChecksOnlyApplicableCatalogDimension(t *testing.T) {
	tests := []struct {
		name       string
		usage      ProfileBudgetUsage
		exhausted  ProfileBudgetDimension
		applicable ProfileBudgetDimension
	}{
		{
			name:       "labels",
			usage:      ProfileBudgetUsage{Labels: 1},
			exhausted:  ProfileBudgetLabels,
			applicable: ProfileBudgetProperties,
		},
		{
			name:       "properties",
			usage:      ProfileBudgetUsage{Properties: 1},
			exhausted:  ProfileBudgetProperties,
			applicable: ProfileBudgetLabels,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			budget := NewProfileBudget(ProfileBudgetLimits{
				Rows: 10, Pages: 10, Labels: 1, Properties: 1,
			})
			if err := budget.Charge(test.usage); err != nil {
				t.Fatal(err)
			}
			if err := budget.Full(test.exhausted); !errors.Is(err, ErrProfileBudget) {
				t.Fatalf("Full(exhausted) error = %v", err)
			}
			if err := budget.Full(test.applicable); err != nil {
				t.Fatalf("Full(applicable) error = %v", err)
			}
			if err := budget.Full(); err != nil {
				t.Fatalf("Full() row-iteration error = %v", err)
			}
			if err := budget.CanProcess(); err != nil {
				t.Fatalf("CanProcess() error = %v", err)
			}
			if err := budget.Charge(ProfileBudgetUsage{Rows: 1}); err != nil {
				t.Fatalf("row charge after catalog exhaustion = %v", err)
			}
		})
	}
}
