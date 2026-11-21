#!/bin/bash
set -euo pipefail
umask 077

[[ "$(hostname -s)" == afhs-20261118-p1-csv && "$(id -u)" == 0 ]]
case "$(pwd -P)" in
  /var/lib/agefreighter-p1/r20261120p1neo44pg17|\
  /var/lib/agefreighter-p1/r20261120p1neo44pg17b|\
  /var/lib/agefreighter-p1/r20261120p1neo44pg18|\
  /var/lib/agefreighter-p1/r20261120p1neo44pg18b)
    container=afh-p1-neo4j44
    image='neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553'
    port=7689
    settings=dbms
    connector=dbms.connector
    admin=(set-initial-password)
    ;;
  /var/lib/agefreighter-p1/r20261121p1neo526pg17|\
  /var/lib/agefreighter-p1/r20261121p1neo526pg17b|\
  /var/lib/agefreighter-p1/r20261121p1neo526pg18)
    container=afh-p1-neo4j526
    image='neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d'
    port=7690
    settings=server
    connector=server
    admin=(dbms set-initial-password)
    ;;
  *) exit 1 ;;
esac
[[ "$(wc -l </proc/swaps)" -eq 1 ]]
[[ "$(findmnt -no FSTYPE --target /dev/shm)" == tmpfs ]]
secrets=/dev/shm/agefreighter-p1-neo4j
data="$PWD/neo4j-data"
conf="$PWD/neo4j-conf"
mounts=(--mount "type=bind,source=$data,target=/data"
        --mount "type=bind,source=$conf,target=/conf,readonly")

verify_loopback_bolt() {
  case "$(ss -H -ltn "sport = :$port" | awk '{print $4}')" in
    "127.0.0.1:$port"|"[::ffff:127.0.0.1]:$port") return 0 ;;
    *) return 1 ;;
  esac
}

case "${1:-start}" in
  start)
    test ! -e neo4j-source-runtime.json
    install -d -m 0700 "$secrets"
    install -d -m 0700 -o 7474 -g 7474 "$data"
    install -d -m 0755 "$conf"
    export DEBIAN_FRONTEND=noninteractive
    timeout --kill-after=10s 3m apt-get -o Acquire::Retries=2 update -qq
    timeout --kill-after=10s 5m apt-get -o Acquire::Retries=2 install -y -qq docker.io
    systemctl enable --now docker
    timeout --kill-after=10s 3m docker pull --quiet "$image"
    cat >"$conf/neo4j.conf" <<CONF
$settings.directories.data=/data
$settings.directories.logs=/data/logs
$settings.default_listen_address=127.0.0.1
$connector.bolt.listen_address=127.0.0.1:$port
$connector.bolt.advertised_address=127.0.0.1:$port
$connector.http.enabled=false
$connector.https.enabled=false
$settings.memory.heap.initial_size=1g
$settings.memory.heap.max_size=3g
$settings.memory.pagecache.size=3g
$settings.jvm.additional=-XX:+ExitOnOutOfMemoryError
CONF
    chmod 0644 "$conf/neo4j.conf"
    password=$(openssl rand -hex 32)
    printf '%s' "$password" >"$secrets/password"
    printf '%s\n' "$password" | docker run --rm -i --network none --user 7474:7474 \
      "${mounts[@]}" --env NEO4J_CONF=/conf --entrypoint /bin/bash "$image" \
      -c 'IFS= read -r password; exec /var/lib/neo4j/bin/neo4j-admin "$@" "$password"' bash "${admin[@]}"
    unset password
    ;;
  verify)
    test ! -e neo4j-source-runtime.json
    test ! -e neo4j-source-frozen.json
    [[ "$(docker inspect --format '{{.Config.Image}} {{.State.Running}} {{.State.OOMKilled}}' "$container")" == "$image true false" ]]
    verify_loopback_bolt
    printf '{"image":"%s","bind":"127.0.0.1:%s"}\n' "$image" "$port" >neo4j-source-runtime.json
    exit 0
    ;;
  freeze)
    test ! -e neo4j-source-frozen.json
    [[ "$(docker inspect --format '{{.Config.Image}} {{.State.Running}} {{.State.OOMKilled}}' "$container")" == "$image true false" ]]
    # Flushing the seeded store and indexes can take longer than Docker's usual stop grace.
    docker stop --time 180 "$container" >/dev/null
    shutdown=$(docker inspect --format '{{.State.ExitCode}} {{.State.OOMKilled}}' "$container")
    if [[ "$shutdown" != "0 false" ]]; then
      echo "Neo4j source did not shut down cleanly" >&2
      exit 1
    fi
    docker rm "$container" >/dev/null
    printf '%s.databases.read_only=neo4j\n' "$settings" >>"$conf/neo4j.conf"
    ;;
  *) exit 1 ;;
esac

test -z "$(ss -H -ltn "sport = :$port")"
docker run --detach --name "$container" --network host --user 7474:7474 \
  --memory 8g --memory-swap 8g --ulimit core=0 --log-opt max-size=16m --log-opt max-file=2 \
  "${mounts[@]}" --env NEO4J_CONF=/conf --entrypoint /var/lib/neo4j/bin/neo4j "$image" console >/dev/null
for _ in $(seq 1 60); do
  if verify_loopback_bolt; then
    break
  fi
  sleep 2
done
verify_loopback_bolt
[[ "$(docker inspect --format '{{.Config.Image}} {{.State.Running}} {{.State.OOMKilled}}' "$container")" == "$image true false" ]]
if [[ "${1:-start}" == freeze ]]; then
  printf '{"image":"%s","bind":"127.0.0.1:%s","readOnlyConfigured":true}\n' "$image" "$port" >neo4j-source-frozen.json
else
  printf '{"image":"%s","bind":"127.0.0.1:%s"}\n' "$image" "$port" >neo4j-source-runtime.json
fi
