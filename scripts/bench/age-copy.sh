#!/bin/sh

set -eu

if [ "$#" -ne 5 ]; then
	printf 'Usage: %s WORKLOAD STRATEGY ROWS TRIALS OUTPUT_JSONL\n' "$0" >&2
	exit 2
fi

workload=$1
strategy=$2
rows=$3
trials=$4
output=$5
container_name=agefreighter-pg17-age160
property_bytes=${BENCH_PROPERTY_BYTES:-64}
endpoint_vertices=${BENCH_ENDPOINT_VERTICES:-10000}
dsn=${AGEFREIGHTER_AGE_TEST_DSN:-}
expected_memory_limit=${BENCH_TARGET_MEMORY_BYTES:-1073741824}

if [ -z "$dsn" ]; then
	printf 'AGEFREIGHTER_AGE_TEST_DSN is required\n' >&2
	exit 1
fi
case "$workload" in
vertices | edges) ;;
*)
	printf 'Unsupported workload: %s\n' "$workload" >&2
	exit 2
	;;
esac
case "$strategy" in
direct-text | staged-binary | plain-relational) ;;
*)
	printf 'Unsupported strategy: %s\n' "$strategy" >&2
	exit 2
	;;
esac
for value in "$rows" "$trials" "$property_bytes" "$endpoint_vertices" "$expected_memory_limit"; do
	case "$value" in
	'' | *[!0-9]* | 0)
		printf 'Rows, trials, property bytes, endpoint vertices, and memory limit must be positive integers\n' >&2
		exit 2
		;;
	esac
done
if [ -e "$output" ]; then
	printf 'Refusing to overwrite benchmark output: %s\n' "$output" >&2
	exit 1
fi
mkdir -p "$(dirname "$output")"

command -v container >/dev/null 2>&1 || {
	printf 'Apple Container CLI is required\n' >&2
	exit 1
}
command -v jq >/dev/null 2>&1 || {
	printf 'jq is required\n' >&2
	exit 1
}

validate_counter() {
	name=$1
	value=$2
	case "$value" in
	'' | *[!0-9]*)
		printf 'Invalid %s cgroup counter: %s\n' "$name" "$value" >&2
		return 1
		;;
	esac
	printf '%s\n' "$value"
}

read_cpu_usage() {
	value=$(container exec "$container_name" cat /sys/fs/cgroup/cpu.stat |
		awk '$1 == "usage_usec" { print $2 }')
	validate_counter cpu.usage_usec "$value"
}

read_memory_counter() {
	name=$1
	value=$(container exec "$container_name" cat "/sys/fs/cgroup/$name")
	validate_counter "$name" "$value"
}

result_file=
error_file=
time_file=
pid_file=
benchmark_pid=
partial_output=
lock_dir="${output}.lock"
lock_acquired=
target_lock_dir="/tmp/${container_name}-$(id -u)-benchmark.lock"
target_lock_acquired=
cleanup_trial() {
	for file in "$result_file" "$error_file" "$time_file" "$pid_file"; do
		if [ -n "$file" ]; then
			rm -f "$file"
		fi
	done
}
cleanup() {
	cleanup_trial
	if [ -n "$partial_output" ]; then
		rm -f "$partial_output"
	fi
	if [ -n "$lock_acquired" ]; then
		rmdir "$lock_dir" || printf 'Failed to remove benchmark lock: %s\n' "$lock_dir" >&2
	fi
	if [ -n "$target_lock_acquired" ]; then
		rmdir "$target_lock_dir" ||
			printf 'Failed to remove benchmark target lock: %s\n' "$target_lock_dir" >&2
	fi
}
forward_signal() {
	signal=$1
	forwarded=
	if [ -n "$benchmark_pid" ] && [ -n "$pid_file" ]; then
		while [ ! -s "$pid_file" ]; do
			if ! kill -0 "$benchmark_pid" 2>/dev/null; then
				break
			fi
			sleep 0.01
		done
	fi
	if [ -n "$pid_file" ] && [ -s "$pid_file" ]; then
		workload_pid=$(cat "$pid_file")
		case "$workload_pid" in
		'' | *[!0-9]*) ;;
		*)
			kill -"$signal" "$workload_pid" 2>/dev/null || :
			forwarded=1
			;;
		esac
	fi
	if [ -n "$benchmark_pid" ]; then
		if [ -z "$forwarded" ]; then
			kill -"$signal" "$benchmark_pid" 2>/dev/null || :
		fi
	fi
}
terminate() {
	signal=$1
	status=$2
	trap 'forward_signal HUP' HUP
	trap 'forward_signal INT' INT
	trap 'forward_signal TERM' TERM
	forward_signal "$signal"
	if [ -n "$benchmark_pid" ]; then
		wait "$benchmark_pid" 2>/dev/null || :
	fi
	exit "$status"
}
trap cleanup EXIT
trap 'terminate HUP 129' HUP
trap 'terminate INT 130' INT
trap 'terminate TERM 143' TERM

if ! mkdir "$target_lock_dir"; then
	printf 'Benchmark target is already in use: %s\n' "$container_name" >&2
	exit 1
fi
target_lock_acquired=1
if ! mkdir "$lock_dir"; then
	printf 'Benchmark output is already reserved: %s\n' "$output" >&2
	exit 1
fi
lock_acquired=1
if [ -e "$output" ]; then
	printf 'Refusing to overwrite benchmark output: %s\n' "$output" >&2
	exit 1
fi
partial_output=$(mktemp "${output}.tmp.XXXXXX")

go build -trimpath -o bin/agefreighter-tools ./cmd/agefreighter-tools

trial=1
while [ "$trial" -le "$trials" ]; do
	container stop "$container_name" >/dev/null
	container start "$container_name" >/dev/null
	ready=0
	while [ "$ready" -lt 90 ]; do
		if container exec "$container_name" \
			pg_isready -U agefreighter -d agefreighter >/dev/null 2>&1; then
			break
		fi
		ready=$((ready + 1))
		sleep 1
	done
	if [ "$ready" -ge 90 ]; then
		printf 'AGE target did not become ready after restart\n' >&2
		exit 1
	fi

	result_file=$(mktemp)
	error_file=$(mktemp)
	time_file=$(mktemp)
	pid_file=$(mktemp)
	before_memory=$(read_memory_counter memory.current)
	before_peak=$(read_memory_counter memory.peak)
	before_limit=$(read_memory_counter memory.max)
	if [ "$before_limit" -ne "$expected_memory_limit" ]; then
		printf 'AGE target memory limit is %s bytes, expected %s\n' \
			"$before_limit" "$expected_memory_limit" >&2
		exit 1
	fi
	# Read CPU last before and first after the workload so observer processes
	# do not run inside the measured interval.
	before_cpu=$(read_cpu_usage)

	# Put the measured process in its own group so terminal signals reach the
	# harness once and are forwarded once.
	set -m
	/usr/bin/time -l -o "$time_file" \
		sh -c 'printf "%s\n" "$$" >"$1"; shift; exec "$@"' \
		agefreighter-benchmark "$pid_file" \
		bin/agefreighter-tools benchmark-age-copy \
		--workload "$workload" \
		--strategy "$strategy" \
		--rows "$rows" \
		--endpoint-vertices "$endpoint_vertices" \
		--property-bytes "$property_bytes" \
		>"$result_file" 2>"$error_file" &
	benchmark_pid=$!
	set +m
	if wait "$benchmark_pid"; then
		benchmark_status=0
	else
		benchmark_status=$?
	fi
	benchmark_pid=
	if [ "$benchmark_status" -ne 0 ]; then
		cat "$error_file" >&2
		exit 1
	fi

	after_cpu=$(read_cpu_usage)
	after_memory=$(read_memory_counter memory.current)
	after_peak=$(read_memory_counter memory.peak)
	if [ "$after_peak" -le "$before_peak" ]; then
		printf 'Workload did not establish a new target cgroup memory peak\n' >&2
		exit 1
	fi
	cpu_usec=$((after_cpu - before_cpu))
	peak_above_baseline=$((after_peak - before_memory))
	client_rss=$(awk '/maximum resident set size/ { print $1 }' "$time_file")

	jq -c \
		--argjson trial "$trial" \
		--argjson targetCPUUsec "$cpu_usec" \
		--argjson baselineTargetCgroupMemoryBytes "$before_memory" \
		--argjson preRunTargetCgroupPeakBytes "$before_peak" \
		--argjson targetMemoryLimitBytes "$before_limit" \
		--argjson postTargetCgroupMemoryBytes "$after_memory" \
		--argjson peakTargetCgroupMemoryBytes "$after_peak" \
		--argjson peakTargetCgroupMemoryAboveBaselineBytes "$peak_above_baseline" \
		--argjson peakClientRSSBytes "$client_rss" \
		'. + {
			trial: $trial,
			targetCPUUsec: $targetCPUUsec,
			baselineTargetCgroupMemoryBytes: $baselineTargetCgroupMemoryBytes,
			preRunTargetCgroupPeakBytes: $preRunTargetCgroupPeakBytes,
			targetMemoryLimitBytes: $targetMemoryLimitBytes,
			postTargetCgroupMemoryBytes: $postTargetCgroupMemoryBytes,
			peakTargetCgroupMemoryBytes: $peakTargetCgroupMemoryBytes,
			peakTargetCgroupMemoryAboveBaselineBytes: $peakTargetCgroupMemoryAboveBaselineBytes,
			peakClientRSSBytes: $peakClientRSSBytes
		}' "$result_file" >>"$partial_output"

	cleanup_trial
	result_file=
	error_file=
	time_file=
	pid_file=
	trial=$((trial + 1))
done

mv -n "$partial_output" "$output"
if [ -e "$partial_output" ]; then
	printf 'Refusing to overwrite benchmark output: %s\n' "$output" >&2
	exit 1
fi
partial_output=
