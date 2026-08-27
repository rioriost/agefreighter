package tools

import (
	"errors"
	"fmt"
	"math"
)

type benchmarkComparisonKey struct {
	workload      BenchmarkWorkload
	rows          int
	propertyBytes int
}

type benchmarkComparison struct {
	staged     *BenchmarkReportGroup
	relational *BenchmarkReportGroup
}

func ValidateStagedBenchmarkBudget(
	report BenchmarkReport,
	minimumRatio float64,
) error {
	if math.IsNaN(minimumRatio) || math.IsInf(minimumRatio, 0) ||
		minimumRatio <= 0 || minimumRatio > 10 {
		return errors.New("minimum staged benchmark ratio must be greater than 0 and at most 10")
	}
	canonical, err := canonicalBenchmarkReport(report)
	if err != nil {
		return err
	}
	comparisons := make(map[benchmarkComparisonKey]benchmarkComparison)
	keys := make([]benchmarkComparisonKey, 0)
	for index := range canonical.Groups {
		group := &canonical.Groups[index]
		if group.Strategy != BenchmarkStaged &&
			group.Strategy != BenchmarkRelational {
			continue
		}
		key := benchmarkComparisonKey{
			workload:      group.Workload,
			rows:          group.Rows,
			propertyBytes: group.PropertyBytes,
		}
		if _, exists := comparisons[key]; !exists {
			keys = append(keys, key)
		}
		comparison := comparisons[key]
		if group.Strategy == BenchmarkStaged {
			comparison.staged = group
		} else {
			comparison.relational = group
		}
		comparisons[key] = comparison
	}
	if len(comparisons) == 0 {
		return errors.New("benchmark report contains no staged/relational comparisons")
	}
	for _, key := range keys {
		comparison := comparisons[key]
		if comparison.staged == nil || comparison.relational == nil {
			return fmt.Errorf(
				"%s benchmark with %d rows and %d property bytes must include both staged-binary and plain-relational samples",
				key.workload,
				key.rows,
				key.propertyBytes,
			)
		}
		actualRatio := comparison.staged.RowsPerSecond.Median /
			comparison.relational.RowsPerSecond.Median
		if actualRatio < minimumRatio {
			return fmt.Errorf(
				"%s staged-binary throughput ratio %.4f is below required %.4f for %d rows and %d property bytes",
				key.workload,
				actualRatio,
				minimumRatio,
				key.rows,
				key.propertyBytes,
			)
		}
	}
	return nil
}
