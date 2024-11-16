# Node Graph Client

A Python client implementation for interacting with the Horsona node graph server. This project provides an example of generating and using client libraries from OpenAPI specifications.

## Setup
1. Start the Node Graph server
```bash
(cd ../../; poetry run python -m horsona.interface.node_graph)
```

2. Create & install the client libraries
```bash
# Install the OpenAPI client generator
poetry install --no-root

# Generate from local server
poetry run python -m openapi_python_client generate --url http://localhost:8000/openapi.json
poetry run python -m openapi_python_client generate --url http://localhost:8000/api/openapi.json

# Set up the generated client libraries as their own projects
poetry -C horsona-node-graph-client/ install
poetry -C horsona-modules-client/ install

# Add local client packages to project
poetry add ./horsona-node-graph-client ./horsona-modules-client
```

## Running the client

```bash
poetry run python src/main.py
```