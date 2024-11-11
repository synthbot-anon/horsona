#!/bin/sh

projroot=$(dirname "$(readlink -f $0)")
cd "$projroot"

poetry run ruff check --select I --fix
poetry run ruff format

