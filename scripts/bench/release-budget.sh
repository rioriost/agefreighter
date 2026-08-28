#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
	printf 'usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
	exit 2
fi
if [ -z "${AGEFREIGHTER_AGE_TEST_DSN:-}" ]; then
	printf 'AGEFREIGHTER_AGE_TEST_DSN is required\n' >&2
	exit 2
fi

output=$1
case "$output" in
	""|-*)
		printf 'output directory must be a non-option path\n' >&2
		exit 2
		;;
esac
if [ -L "$output" ]; then
	printf 'output directory must not be a symbolic link\n' >&2
	exit 2
fi
samples=${AGEFREIGHTER_BENCH_SAMPLES:-3}
rows=${AGEFREIGHTER_BENCH_ROWS:-100000}
property_bytes=${AGEFREIGHTER_BENCH_PROPERTY_BYTES:-64}
minimum_ratio=${AGEFREIGHTER_MINIMUM_STAGED_RATIO:-0.40}
minimum_csv_rows_per_second=${AGEFREIGHTER_MINIMUM_CSV_ROWS_PER_SECOND:-109190}

for value in "$samples" "$rows" "$property_bytes"; do
	if ! printf '%s\n' "$value" | awk 'BEGIN { ok = 0 } /^[0-9]+$/ { ok = ($0 + 0 > 0) } END { exit !ok }'; then
		printf 'sample, row, and property-byte values must be positive integers\n' >&2
		exit 2
	fi
done
for value in "$minimum_ratio" "$minimum_csv_rows_per_second"; do
	if ! printf '%s\n' "$value" | awk 'BEGIN { ok = 0 } /^[0-9]+([.][0-9]+)?$/ { ok = ($0 + 0 > 0) } END { exit !ok }'; then
		printf 'performance thresholds must be positive numbers\n' >&2
		exit 2
	fi
done

mkdir -p "$output"
raw="$output/age-copy.jsonl"
report="$output/age-copy-report.json"
csv_output="$output/csv-create.txt"
binary="$output/.agefreighter-tools"
trap 'rm -f "$binary"' EXIT HUP INT TERM
: >"$raw"

go build -trimpath -o "$binary" ./cmd/agefreighter-tools
for workload in vertices edges; do
	for strategy in staged-binary plain-relational; do
		sample=1
		while [ "$sample" -le "$samples" ]; do
			"$binary" benchmark-age-copy \
				--workload "$workload" \
				--strategy "$strategy" \
				--rows "$rows" \
				--property-bytes "$property_bytes" >>"$raw"
			sample=$((sample + 1))
		done
	done
done
"$binary" benchmark-report \
	--minimum-staged-ratio "$minimum_ratio" \
	"$raw" >"$report"

AGEFREIGHTER_AGE_TEST_DSN="$AGEFREIGHTER_AGE_TEST_DSN" \
	go test -run '^$' -bench '^BenchmarkLegacyCountriesLoad$' \
	-benchtime="${samples}x" -count=1 ./internal/app | tee "$csv_output"

csv_rows_per_second=$(awk '
	/^BenchmarkLegacyCountriesLoad-/ {
		for (field = 1; field < NF; field++) {
			if ($(field + 1) == "rows/s") {
				value = $field
			}
		}
	}
	END {
		if (value == "") {
			exit 1
		}
		print value
	}
' "$csv_output") || {
	printf 'CSV rows/s metric not found in benchmark output\n' >&2
	exit 1
}
if ! awk -v actual="$csv_rows_per_second" -v minimum="$minimum_csv_rows_per_second" \
	'BEGIN { exit !(actual + 0 >= minimum + 0) }'; then
	printf 'CSV throughput %s rows/s is below required %s rows/s\n' \
		"$csv_rows_per_second" "$minimum_csv_rows_per_second" >&2
	exit 1
fi
printf 'CSV throughput %s rows/s meets required %s rows/s\n' \
	"$csv_rows_per_second" "$minimum_csv_rows_per_second"
