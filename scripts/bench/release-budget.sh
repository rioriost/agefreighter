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
maximum_client_rss_bytes=${AGEFREIGHTER_MAXIMUM_CLIENT_RSS_BYTES:-2147483648}
memory_rows=${AGEFREIGHTER_MEMORY_GATE_ROWS:-200000}

for value in "$samples" "$rows" "$property_bytes" "$maximum_client_rss_bytes" "$memory_rows"; do
	if ! printf '%s\n' "$value" | awk 'BEGIN { ok = 0 } /^[0-9]+$/ { ok = ($0 + 0 > 0) } END { exit !ok }'; then
		printf 'sample, row, property-byte, RSS, and memory-row values must be positive integers\n' >&2
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

run_with_rss_gate() {
	metric_file=$1
	transcript=$2
	shift 2
	time_file="${metric_file}.time"
	case "$(uname -s)" in
	Darwin)
		/usr/bin/time -l -o "$time_file" "$@" >"$transcript" 2>&1
		rss_bytes=$(awk '/maximum resident set size/ { print $1 }' "$time_file")
		;;
	Linux)
		/usr/bin/time -v -o "$time_file" "$@" >"$transcript" 2>&1
		rss_kib=$(awk -F: '/Maximum resident set size/ {
			gsub(/[[:space:]]/, "", $2); print $2
		}' "$time_file")
		rss_bytes=$((rss_kib * 1024))
		;;
	*)
		printf 'unsupported platform for RSS gate: %s\n' "$(uname -s)" >&2
		return 2
		;;
	esac
	rm -f "$time_file"
	case "$rss_bytes" in
	""|*[!0-9]*)
		printf 'maximum resident set size was not reported\n' >&2
		return 1
		;;
	esac
	printf '{"maximumClientRSSBytes":%s,"limitBytes":%s}\n' \
		"$rss_bytes" "$maximum_client_rss_bytes" >"$metric_file"
	if [ "$rss_bytes" -gt "$maximum_client_rss_bytes" ]; then
		printf 'client RSS %s bytes exceeds required maximum %s bytes\n' \
			"$rss_bytes" "$maximum_client_rss_bytes" >&2
		return 1
	fi
	printf 'client RSS %s bytes meets required maximum %s bytes\n' \
		"$rss_bytes" "$maximum_client_rss_bytes"
}

go build -trimpath -o "$binary" ./cmd/agefreighter-tools
for workload in vertices edges; do
	sample=1
	while [ "$sample" -le "$samples" ]; do
		if [ $((sample % 2)) -eq 1 ]; then
			strategies="staged-binary plain-relational"
		else
			strategies="plain-relational staged-binary"
		fi
		for strategy in $strategies; do
			"$binary" benchmark-age-copy \
				--workload "$workload" \
				--strategy "$strategy" \
				--rows "$rows" \
				--property-bytes "$property_bytes" >>"$raw"
		done
		sample=$((sample + 1))
	done
done
"$binary" benchmark-report \
	--minimum-staged-ratio "$minimum_ratio" \
	"$raw" >"$report"

run_with_rss_gate \
	"$output/csv-create-rss.json" \
	"$csv_output" \
	env AGEFREIGHTER_AGE_TEST_DSN="$AGEFREIGHTER_AGE_TEST_DSN" \
	go test -run '^$' -bench '^BenchmarkLegacyCountriesLoad$' \
	-benchtime="${samples}x" -count=1 ./internal/app
cat "$csv_output"

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

memory_output="$output/memory-scale.txt"
run_with_rss_gate \
	"$output/memory-scale-rss.json" \
	"$memory_output" \
	env AGEFREIGHTER_AGE_TEST_DSN="$AGEFREIGHTER_AGE_TEST_DSN" \
	AGEFREIGHTER_BENCH_ROWS="$memory_rows" \
	go test -run '^$' -bench '^BenchmarkGeneratedCSVLoad$' \
	-benchtime='1x' -count=1 ./internal/app
cat "$memory_output"

if ! awk -v actual="$csv_rows_per_second" -v minimum="$minimum_csv_rows_per_second" \
	'BEGIN { exit !(actual + 0 >= minimum + 0) }'; then
	printf 'CSV throughput %s rows/s is below required %s rows/s\n' \
		"$csv_rows_per_second" "$minimum_csv_rows_per_second" >&2
	exit 1
fi
printf 'CSV throughput %s rows/s meets required %s rows/s\n' \
	"$csv_rows_per_second" "$minimum_csv_rows_per_second"
