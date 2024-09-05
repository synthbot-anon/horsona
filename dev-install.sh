#!/bin/bash

projroot=$(dirname "$(readlink -f $0)")
cd "$projroot"

rm poetry.lock
poetry install

# Check if .env file exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Make sure to configure .env before running the project."
fi
