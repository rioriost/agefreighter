#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 1 ]; then
	printf 'usage: %s P0|P1|P2|P3 phase in lowercase\n' "$0" >&2
	exit 2
fi

phase=$1
case "$phase" in
	p0|p1|p2|p3) ;;
	*)
		printf 'unsupported phase: %s\n' "$phase" >&2
		exit 2
		;;
esac

require_phase_approval "$phase"
require_command psql
require_nonempty_environment AGEFREIGHTER_ADMIN_DSN

database44="agefreighter_${phase}_neo4j44"
database526="agefreighter_${phase}_neo4j526"

for database in "$database44" "$database526"; do
	exists=$(psql "$AGEFREIGHTER_ADMIN_DSN" -X -A -t \
		--set ON_ERROR_STOP=1 \
		--command "SELECT 1 FROM pg_database WHERE datname = '$database'")
	if [ -n "$exists" ]; then
		printf 'target database already exists; refusing reuse: %s\n' "$database" >&2
		exit 3
	fi
done

for database in "$database44" "$database526"; do
	psql "$AGEFREIGHTER_ADMIN_DSN" -X --set ON_ERROR_STOP=1 \
		--command "CREATE DATABASE $database" >/dev/null
	psql "$AGEFREIGHTER_ADMIN_DSN dbname=$database" -X --set ON_ERROR_STOP=1 \
		--command 'CREATE EXTENSION age' >/dev/null
	actual=$(psql "$AGEFREIGHTER_ADMIN_DSN dbname=$database" -X -A -t -F '|' \
		--set ON_ERROR_STOP=1 \
		--command "SELECT current_setting('server_version_num')::integer / 10000, extversion FROM pg_extension WHERE extname = 'age'")
	if [ "$actual" != '18|1.7' ] && [ "$actual" != '18|1.7.0' ]; then
		printf 'target version gate failed for %s: %s\n' "$database" "$actual" >&2
		exit 4
	fi
	printf 'prepared isolated target database: %s (PostgreSQL 18 / AGE 1.7.x)\n' "$database"
done
