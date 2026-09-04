#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

require_command go
require_command shellcheck

repo=$(repository_root)
simulation=$(simulation_root)
cd "$repo"

go test ./production-simulation/...
go run ./cmd/agefreighter validate "$simulation/configs/neo4j-4.4.48.yaml"
go run ./cmd/agefreighter validate "$simulation/configs/neo4j-5.26.30.yaml"
shellcheck -x -P SCRIPTDIR "$simulation"/scripts/*.sh

if command -v az >/dev/null 2>&1; then
	az bicep build --file "$simulation/infra/main.bicep" --stdout >/dev/null
	az bicep build --file "$simulation/infra/horizondb.bicep" --stdout >/dev/null
	az bicep build --file "$simulation/infra/horizondb-private-endpoint.bicep" --stdout >/dev/null
else
	printf 'warning: Azure CLI is absent; skipped Bicep compilation\n' >&2
fi
