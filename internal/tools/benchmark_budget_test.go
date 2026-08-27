package tools

import (
	"math"
	"strings"
	"testing"
)

func TestValidateStagedBenchmarkBudget(t *testing.T) {
	report := BenchmarkReport{
		SchemaVersion: BenchmarkReportSchemaVersion,
		Groups: []BenchmarkReportGroup{
			budgetGroup(BenchmarkVertices, BenchmarkStaged, 80),
			budgetGroup(BenchmarkVertices, BenchmarkRelational, 100),
			budgetGroup(BenchmarkVertices, BenchmarkDirect, 20),
			budgetGroup(BenchmarkEdges, BenchmarkStaged, 70),
			budgetGroup(BenchmarkEdges, BenchmarkRelational, 100),
		},
	}
	if err := ValidateStagedBenchmarkBudget(report, 0.70); err != nil {
		t.Fatalf("ValidateStagedBenchmarkBudget() error = %v", err)
	}
	if err := ValidateStagedBenchmarkBudget(report, 0.71); err == nil ||
		!strings.Contains(err.Error(), "edges") {
		t.Fatalf("failing budget error = %v", err)
	}
}

func TestValidateStagedBenchmarkBudgetRejectsInvalidReports(t *testing.T) {
	valid := BenchmarkReport{
		SchemaVersion: BenchmarkReportSchemaVersion,
		Groups: []BenchmarkReportGroup{
			budgetGroup(BenchmarkVertices, BenchmarkStaged, 80),
			budgetGroup(BenchmarkVertices, BenchmarkRelational, 100),
		},
	}
	for _, ratio := range []float64{-1, 0, 11, math.NaN(), math.Inf(1)} {
		if err := ValidateStagedBenchmarkBudget(valid, ratio); err == nil {
			t.Fatalf("ratio %v accepted", ratio)
		}
	}

	noComparisons := valid
	noComparisons.Groups = []BenchmarkReportGroup{
		budgetGroup(BenchmarkVertices, BenchmarkDirect, 80),
	}
	if err := ValidateStagedBenchmarkBudget(noComparisons, 0.7); err == nil {
		t.Fatal("report without comparisons accepted")
	}

	unmatched := valid
	unmatched.Groups = []BenchmarkReportGroup{
		budgetGroup(BenchmarkEdges, BenchmarkStaged, 80),
	}
	if err := ValidateStagedBenchmarkBudget(unmatched, 0.7); err == nil ||
		!strings.Contains(err.Error(), "both") {
		t.Fatalf("unmatched report error = %v", err)
	}

	invalid := valid
	invalid.SchemaVersion = 2
	if err := ValidateStagedBenchmarkBudget(invalid, 0.7); err == nil {
		t.Fatal("invalid report accepted")
	}
}

func budgetGroup(
	workload BenchmarkWorkload,
	strategy BenchmarkStrategy,
	throughput float64,
) BenchmarkReportGroup {
	return BenchmarkReportGroup{
		Workload: workload, Rows: 1_000, PropertyBytes: 64,
		Strategy: strategy, SampleCount: 3,
		ElapsedNanos: BenchmarkIntegerSummary{
			Median: 1_000_000_000,
			Min:    900_000_000,
			Max:    1_100_000_000,
		},
		RowsPerSecond: BenchmarkFloatSummary{
			Median: throughput,
			Min:    throughput * 0.9,
			Max:    throughput * 1.1,
		},
		WALBytes: BenchmarkIntegerSummary{Median: 100, Min: 90, Max: 110},
	}
}
