#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 1 ]; then
	printf 'usage: %s PHASE\n' "$0" >&2
	exit 2
fi

phase=$1
case "$phase" in
	p0|p1|p2|p3) ;;
	*)
		printf 'unsupported live phase: %s\n' "$phase" >&2
	exit 2
	;;
esac

require_phase_approval "$phase"
require_command psql
require_nonempty_environment AGEFREIGHTER_TARGET_DSN

attempt=0
while :; do
	if actual=$(psql "$AGEFREIGHTER_TARGET_DSN" -X -A -t -F '|' \
		--set ON_ERROR_STOP=1 \
		--command "SELECT current_setting('server_version_num')::integer / 10000, extversion FROM pg_extension WHERE extname = 'age'" \
		2>/dev/null); then
		break
	fi
	attempt=$((attempt + 1))
	if [ "$attempt" -ge 60 ]; then
		printf 'target SQL readiness gate failed after %d attempts\n' "$attempt" >&2
		exit 3
	fi
	sleep 5
done

if [ "$actual" != '18|1.7' ] && [ "$actual" != '18|1.7.0' ]; then
	printf 'target version gate failed: expected PostgreSQL 18 / AGE 1.7.x, received %s\n' "${actual:-<no AGE extension>}" >&2
	exit 4
fi

printf 'target version gate passed: PostgreSQL 18 / Apache AGE 1.7.x\n'
