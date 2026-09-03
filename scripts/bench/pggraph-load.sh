#!/bin/sh

set -eu

if [ "$#" -ne 3 ]; then
	printf 'Usage: %s small|medium|production TRIALS OUTPUT\n' "$0" >&2
	exit 2
fi

profile=$1
trials=$2
output=$3
case "$profile" in
small)
	vertices=10000
	edges=25000
	;;
medium)
	vertices=100000
	edges=250000
	;;
production)
	vertices=160000000
	edges=400000000
	if [ "${PGGRAPH_BENCHMARK_PRODUCTION_ACK:-}" != "160000000-400000000" ]; then
		printf 'Production profile requires PGGRAPH_BENCHMARK_PRODUCTION_ACK=160000000-400000000\n' >&2
		exit 2
	fi
	;;
*)
	printf 'Unsupported property graph benchmark profile: %s\n' "$profile" >&2
	exit 2
	;;
esac
case "$trials" in
'' | *[!0-9]* | 0)
	printf 'TRIALS must be a positive integer\n' >&2
	exit 2
	;;
esac
if [ -z "${AGEFREIGHTER_PGGRAPH_TEST_DSN:-}" ]; then
	printf 'AGEFREIGHTER_PGGRAPH_TEST_DSN is required\n' >&2
	exit 1
fi
if [ -e "$output" ]; then
	printf 'Refusing to overwrite benchmark output: %s\n' "$output" >&2
	exit 1
fi
mkdir -p "$(dirname "$output")"

lock_dir="${output}.lock"
partial_output=
cleanup() {
	if [ -n "$partial_output" ]; then
		rm -f "$partial_output"
	fi
	rmdir "$lock_dir" 2>/dev/null || :
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if ! mkdir "$lock_dir"; then
	printf 'Benchmark output is already reserved: %s\n' "$output" >&2
	exit 1
fi
partial_output=$(mktemp "${output}.tmp.XXXXXX")

{
	printf 'profile=%s vertices=%s edges=%s trials=%s\n' \
		"$profile" "$vertices" "$edges" "$trials"
	go version
	./scripts/dev/pggraph-apple-container.sh status
	AGEFREIGHTER_PGGRAPH_BENCHMARK_VERTICES="$vertices" \
	AGEFREIGHTER_PGGRAPH_BENCHMARK_EDGES="$edges" \
		go test ./internal/app -run '^$' \
		-bench '^BenchmarkPostgreSQLPropertyGraphCreate$' \
		-benchtime="${trials}x" -count=1
} >"$partial_output"

mv "$partial_output" "$output"
partial_output=
