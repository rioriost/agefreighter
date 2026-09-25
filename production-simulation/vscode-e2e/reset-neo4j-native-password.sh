#!/usr/bin/env bash
# Recover the native neo4j administrator password without exposing it in
# command arguments or output. Run through Azure Managed Run Command with
# AF_TEMP_PASSWORD supplied as a protected parameter.
set -euo pipefail
set +x
umask 077

container="${AF_NEO4J_CONTAINER:-af-n526-neo4j526}"
expected_vertices="${AF_EXPECTED_VERTICES:-1600000}"
expected_edges="${AF_EXPECTED_EDGES:-4000000}"
recovery=""
original_was_stopped=0
success=0

if [[ ! "${AF_TEMP_PASSWORD:-}" =~ ^[[:xdigit:]]{64}$ ]]; then
  printf '%s\n' '{"status":"failed","stage":"validate-protected-password"}'
  exit 2
fi
if [[ ! "$expected_vertices" =~ ^[0-9]+$ || ! "$expected_edges" =~ ^[0-9]+$ ]]; then
  printf '%s\n' '{"status":"failed","stage":"validate-expected-counts"}'
  exit 2
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null)" != "true" ]]; then
  printf '%s\n' '{"status":"failed","stage":"require-running-original"}'
  exit 3
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
evidence_root="/var/lib/agefreighter/neo4j-password-recovery/$stamp"
install -d -m 0700 "$evidence_root"
image_id="$(docker inspect --format '{{.Image}}' "$container")"
data_dir="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}' "$container")"
if [[ -z "$image_id" || -z "$data_dir" || ! -d "$data_dir/databases/system" ]]; then
  printf '%s\n' '{"status":"failed","stage":"inspect-original"}'
  exit 4
fi
docker inspect "$container" | sha256sum | awk '{print $1}' >"$evidence_root/original-inspect.sha256"

cleanup() {
  local rc=$?
  set +e
  if [[ -n "$recovery" ]] && docker inspect "$recovery" >/dev/null 2>&1; then
    docker inspect --format 'status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{json .State.Error}} network={{.HostConfig.NetworkMode}} ports={{json .HostConfig.PortBindings}}' "$recovery" >"$evidence_root/recovery-container-state.txt" 2>&1
    docker logs "$recovery" >"$evidence_root/recovery-container.log" 2>&1
    docker rm --force "$recovery" >/dev/null 2>&1
  fi
  if [[ "$original_was_stopped" == 1 && "$(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null)" != "true" ]]; then
    docker start "$container" >/dev/null 2>&1
  fi
  if [[ "$success" != 1 ]]; then
    printf '{"status":"failed","stage":"password-recovery","evidence":"%s"}\n' "$stamp"
  fi
  exit "$rc"
}
trap cleanup EXIT

docker stop --time 120 "$container" >/dev/null
original_was_stopped=1

backup_paths=(databases/system)
if [[ -d "$data_dir/transactions/system" ]]; then
  backup_paths+=(transactions/system)
fi
tar --xattrs --numeric-owner -czf "$evidence_root/system-database-before.tgz" -C "$data_dir" "${backup_paths[@]}"
sha256sum "$evidence_root/system-database-before.tgz" >"$evidence_root/system-database-before.tgz.sha256"

recovery="af-n526-password-recovery-${stamp,,}"
docker run --detach \
  --name "$recovery" \
  --network bridge \
  --restart no \
  --volume "$data_dir:/data" \
  --env NEO4J_AUTH=none \
  --env NEO4J_dbms_security_auth__enabled=false \
  --env NEO4J_server_default__listen__address=127.0.0.1 \
  --env NEO4J_server_bolt_listen__address=127.0.0.1:7687 \
  --env NEO4J_server_http_enabled=false \
  --env NEO4J_server_https_enabled=false \
  "$image_id" >/dev/null

# The recovery server has no host port binding and only listens on its own
# loopback interface. Bridge networking avoids assumptions in Neo4j's startup
# scripts while still satisfying the official no-remote-access requirement.
if [[ -n "$(docker port "$recovery")" ]]; then
  exit 5
fi

ready=0
for attempt in $(seq 1 90); do
  if [[ "$(docker inspect --format '{{.State.Running}}' "$recovery")" != "true" ]]; then
    exit 5
  fi
  if docker exec "$recovery" cypher-shell -a bolt://127.0.0.1:7687 -d system --format plain 'SHOW USERS' >"$evidence_root/recovery-probe-last.txt" 2>&1; then
    ready=1
    break
  fi
  if [[ "$attempt" -ge 3 ]] && grep -q 'Started\.' "$evidence_root/recovery-container.log" 2>/dev/null; then
    exit 5
  fi
  docker logs "$recovery" >"$evidence_root/recovery-container.log" 2>&1
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  exit 5
fi

# The password is sent only over stdin to cypher-shell. The hexadecimal format
# enforced above makes the quoted Cypher literal unambiguous.
printf "ALTER USER neo4j SET PASSWORD '%s' CHANGE NOT REQUIRED;\n" "$AF_TEMP_PASSWORD" |
  docker exec --interactive "$recovery" cypher-shell -a bolt://127.0.0.1:7687 -d system --format plain >/dev/null

docker logs "$recovery" >"$evidence_root/recovery-container.log" 2>&1
docker rm --force "$recovery" >/dev/null
recovery=""
docker start "$container" >/dev/null

cypher_query() {
  local query=$1
  printf '%s\n' "$AF_TEMP_PASSWORD" |
    docker exec --interactive "$container" sh -c '
      IFS= read -r NEO4J_PASSWORD
      export NEO4J_PASSWORD NEO4J_USERNAME=neo4j
      exec cypher-shell -a bolt+ssc://127.0.0.1:7687 -d neo4j --format plain "$1"
    ' sh "$query"
}

authenticated=0
for _ in $(seq 1 90); do
  if cypher_query 'RETURN 1 AS ok' >/dev/null 2>&1; then
    authenticated=1
    break
  fi
  sleep 2
done
if [[ "$authenticated" != 1 ]]; then
  exit 6
fi

vertices="$(cypher_query 'MATCH (n) RETURN count(n) AS count' | tail -n 1 | tr -d '[:space:]')"
edges="$(cypher_query 'MATCH ()-[r]->() RETURN count(r) AS count' | tail -n 1 | tr -d '[:space:]')"
if [[ "$vertices" != "$expected_vertices" || "$edges" != "$expected_edges" ]]; then
  exit 7
fi

backup_sha="$(cut -d ' ' -f 1 "$evidence_root/system-database-before.tgz.sha256")"
cat >"$evidence_root/summary.json" <<EOF
{"status":"passed","completedAt":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","container":"$container","imageId":"$image_id","systemBackupSHA256":"$backup_sha","vertices":"$vertices","edges":"$edges","recoveryNetwork":"unpublished-loopback-only","authenticationRestored":true}
EOF
chmod 0600 "$evidence_root/summary.json"
sha256sum "$evidence_root/summary.json" >"$evidence_root/summary.json.sha256"

success=1
printf '{"status":"passed","evidence":"%s","vertices":"%s","edges":"%s"}\n' "$stamp" "$vertices" "$edges"
