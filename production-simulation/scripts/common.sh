#!/bin/sh

set -eu

simulation_root() {
	CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd
}

repository_root() {
	CDPATH='' cd -- "$(simulation_root)/.." && pwd
}

require_command() {
	command -v "$1" >/dev/null 2>&1 || {
		printf 'required command not found: %s\n' "$1" >&2
		exit 2
	}
}

require_phase_approval() {
	phase=$1
	expected="reviewed-$phase"
	if [ "${PRODUCTION_SIMULATION_APPROVAL:-}" != "$expected" ]; then
		printf 'live operation blocked: set PRODUCTION_SIMULATION_APPROVAL=%s after review\n' "$expected" >&2
		exit 3
	fi
}

require_file() {
	if [ ! -f "$1" ]; then
		printf 'required file not found: %s\n' "$1" >&2
		exit 2
	fi
}

require_nonempty_environment() {
	variable_name=$1
	eval "variable_value=\${$variable_name:-}"
	if [ -z "$variable_value" ]; then
		printf 'required environment variable is empty: %s\n' "$variable_name" >&2
		exit 2
	fi
}
