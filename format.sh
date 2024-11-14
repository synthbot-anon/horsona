#!/bin/sh

if printf '%s\n' "$@" | grep -q '[[:space:]]'; then
    echo "Error: Filenames cannot contain whitespace" >&2
    exit 1
fi

projroot=$(dirname "$(readlink -f $0)")
echo $projroot > ~/123
cd "$projroot"

pyfiles=$(printf '%s\n' "$@" | grep '\.py$')

if [ -z "$pyfiles" ] && [ $# -ne 0 ]; then
    exit 0
fi

poetry run ruff check --select I --fix $pyfiles
poetry run ruff format --respect-gitignore $pyfiles

