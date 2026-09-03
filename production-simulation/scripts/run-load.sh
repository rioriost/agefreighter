#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 2 ]; then
	printf 'usage: %s PHASE neo4j-4.4.48|neo4j-5.26.30\n' "$0" >&2
	exit 2
fi

phase=$1
source_version=$2
case "$phase" in
	p0|p1|p2|p3) ;;
	*)
		printf 'live loads support p0, p1, p2, or p3\n' >&2
		exit 2
		;;
esac
case "$source_version" in
	neo4j-4.4.48)
	require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J44
	AGEFREIGHTER_TARGET_DSN=$AGEFREIGHTER_TARGET_DSN_NEO4J44
	;;
	neo4j-5.26.30)
	require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J526
	AGEFREIGHTER_TARGET_DSN=$AGEFREIGHTER_TARGET_DSN_NEO4J526
	;;
	*)
		printf 'unsupported source version: %s\n' "$source_version" >&2
		exit 2
		;;
esac

require_phase_approval "$phase"
require_command agefreighter
require_nonempty_environment AGEFREIGHTER_NEO4J_PASSWORD
export AGEFREIGHTER_TARGET_DSN

config="$(simulation_root)/configs/$source_version.yaml"
require_file "$config"

"$(simulation_root)/scripts/preflight-target.sh" "$phase"
agefreighter validate "$config"
agefreighter plan "$config"
printf 'Starting reviewed %s load from %s. Record the emitted job ID.\n' "$phase" "$source_version" >&2
exec agefreighter load "$config"
