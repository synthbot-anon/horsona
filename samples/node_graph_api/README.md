# Node Graph API Sample Project

This sample project demonstrates how to create a REST API to support remote execution of Horsona modules using FastAPI. The Node Graph API allows clients to create sessions, manage resources, and execute Horsona modules remotely.

## Overview

The main components of this sample project are:

1. `__main__.py`: The entry point of the application that sets up and runs the FastAPI server.
2. `node_graph_api.py`: Contains the `NodeGraphAPI` class, which defines the API endpoints and handles the execution of Horsona modules.
3. `test_node_graph.py`: Provides example usage and tests for the Node Graph API.

## Features

- Create and manage sessions
- Execute Horsona modules remotely
- Support for various argument types, including nested Horsona objects
- Flexible module allowlist configuration

## Usage

To run the Node Graph API server:

1. Clone the repo and navigate to the `node_graph_api/bin/windows` folder. (On Linux: `node_graph_api/bin/linux`)

   ```powershell
   git clone https://github.com/synthbot-anon/horsona.git
   cd horsona/samples/node_graph_api/bin/windows
   # On Linux: cd horsona/samples/node_graph_api/bin/linux
   ```

2. Run `node_graph_api.bat`. (On Linux: `node_graph_api.sh`)

   This will pull the latest Docker image and start the server.

3. The server will start running on `http://localhost:8000`.

To update the Docker image to the latest version:

1. Navigate to the `node_graph_api/bin/windows` folder. (On Linux: `node_graph_api/bin/linux`)
2. Run `update - node_graph_api.bat`. (On Linux: `update - node_graph_api.sh`)

To uninstall the Docker image:

1. Navigate to the `node_graph_api/bin/windows` folder. (On Linux: `node_graph_api/bin/linux`)
2. Run `uninstall - node_graph_api.bat`. (On Linux: `uninstall - node_graph_api.sh`)

## Example: Running node_graph_api.exposed_module.hello_world

Here's an example of how to use curl commands to run the `hello_world` function from the `exposed_module`:

1. Create a new session and get the session ID:

   ```bash
   SESSION_ID=$(curl -X POST http://localhost:8000/api/sessions | jq -r '.session_id')
   ```

2. Run the `hello_world` function:

   ```bash
   curl -X 'POST' \
     'http://localhost:8000/api/sessions/$SESSION_ID/resources' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "module_name": "node_graph_api.exposed_module",
       "function_name": "hello_world",
       "kwargs": {}
     }'
   ```

   This should return a response containing the result "Hello, world!".


3. Delete the session:

   ```bash
   curl -X 'DELETE' \
     "http://localhost:8000/api/sessions/$SESSION_ID" \
     -H 'accept: application/json'
   ```

   This will delete the session and all its associated resources.

## Adding in config files

`node_graph_api.bat` and `node_graph_api.sh` will map the directory they're run from into the container. This allows to to map in configuration files like `.env`, `llm_config.json` and `index_config.json`. Make sure to include all necessary configuration files in the directory you're running from.
