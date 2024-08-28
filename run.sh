#!/bin/bash

projroot=$(dirname "$(readlink -f $0)")
poetry run -C "$projroot" python "$projroot/src/main.py"

