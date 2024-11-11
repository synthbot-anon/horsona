#!/bin/bash

projroot=$(dirname "$(readlink -f $0)")
cd "$projroot"

poetry install

# Check if .env file exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Make sure to configure .env before running the project."
fi

if [ ! -f llm_config.json ]; then
    cp llm_config.json.example llm_config.json
    echo "The default LLM API (OpenAI) is slow. Check the README for how to use other LLM APIs."
fi

if [ ! -f index_config.json ]; then
    cp index_config.json.example index_config.json
    echo "The default embedding API (OpenAI) is slow. Check the README for how to use Ollama embeddings."
fi

echo "Installing pre-commit hooks"
pre-commit install
