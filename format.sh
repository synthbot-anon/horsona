#!/bin/bash

projroot=$(dirname "$(readlink -f $0)")
cd "$projroot"

ruff check --select I --fix
ruff format

