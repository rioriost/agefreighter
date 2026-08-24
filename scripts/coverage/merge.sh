#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
	printf 'usage: %s INPUT_DIRS MERGED_DIR OUTPUT_PROFILE\n' "$0" >&2
	exit 2
fi

input_dirs=$1
merged_dir=$2
output_profile=$3

if [ -e "$merged_dir" ]; then
	printf 'merged coverage directory must not already exist: %s\n' "$merged_dir" >&2
	exit 1
fi

mkdir "$merged_dir"
go tool covdata merge -i="$input_dirs" -o="$merged_dir"
go tool covdata textfmt -i="$merged_dir" -o="$output_profile"
