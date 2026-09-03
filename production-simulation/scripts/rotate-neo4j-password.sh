#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$(id -u)" -ne 0 ]; then
	printf 'run as root on an Azure VM with Key Vault access\n' >&2
	exit 2
fi

require_phase_approval p0
: "${KEY_VAULT_NAME:?KEY_VAULT_NAME is required}"
require_command curl
require_command openssl
require_command python3

token=$(curl -fsS -H Metadata:true \
	'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
neo4j_password=$(openssl rand -hex 32)
body=$(NEO4J_PASSWORD="$neo4j_password" python3 -c \
	'import json,os; print(json.dumps({"value": os.environ["NEO4J_PASSWORD"]}))')

status=$(curl -fsS -o /dev/null -w '%{http_code}' -X PUT \
	-H "Authorization: Bearer $token" \
	-H 'Content-Type: application/json' \
	--data "$body" \
	"https://$KEY_VAULT_NAME.vault.azure.net/secrets/neo4j-password?api-version=7.4")

if [ "$status" != 200 ]; then
	printf 'Key Vault rejected the Neo4j password rotation (HTTP %s)\n' "$status" >&2
	exit 3
fi

unset body neo4j_password token
printf 'Neo4j password rotated in Key Vault without emitting its value\n'
