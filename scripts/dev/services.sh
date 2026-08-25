#!/bin/sh

AGEFREIGHTER_DEV_PASSWORD=agefreighter_dev_only
export AGEFREIGHTER_DEV_PASSWORD

AGE_IMAGE='apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be'
POSTGRES_IMAGE='postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94'
NEO4J_IMAGE='neo4j@sha256:89d577f2e49606de76441eca8cf7a0fe88e594cbaac4d2a3d86c6e59676e2b1e'

AGE_CONTAINER=agefreighter-pg17-age160
POSTGRES_CONTAINER=agefreighter-postgres17-source
NEO4J_CONTAINER=agefreighter-neo4j526

AGE_CONTAINER_CONFIG=age160-pg17-memory1g-v1
POSTGRES_CONTAINER_CONFIG=postgres176-v1
NEO4J_CONTAINER_CONFIG=neo4j526-v1

AGE_VOLUME=agefreighter-pg17-age160-data
POSTGRES_VOLUME=agefreighter-postgres17-source-data
NEO4J_VOLUME=agefreighter-neo4j526-data

AGE_PORT=55432
POSTGRES_PORT=55433
NEO4J_BOLT_PORT=57687
NEO4J_HTTP_PORT=57474

service_containers() {
	printf '%s\n' \
		"$AGE_CONTAINER" \
		"$POSTGRES_CONTAINER" \
		"$NEO4J_CONTAINER"
}

service_volumes() {
	printf '%s\n' \
		"$AGE_VOLUME" \
		"$POSTGRES_VOLUME" \
		"$NEO4J_VOLUME"
}

service_images() {
	printf '%s\n' \
		"$AGE_IMAGE" \
		"$POSTGRES_IMAGE" \
		"$NEO4J_IMAGE"
}

validate_resource_name() {
	case "$1" in
		agefreighter-[a-z0-9]*)
			case "$1" in
				*[!a-z0-9_.-]*)
					return 1
					;;
			esac
			;;
		*)
			return 1
			;;
	esac
}
