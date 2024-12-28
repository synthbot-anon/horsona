# Node Graph API Sample Project

This sample project demonstrates how to create a REST API to support remote execution of Horsona modules using FastAPI. The Node Graph API allows clients to create sessions, manage resources, and execute Horsona modules remotely.

## Overview

The main components of this sample project are:

1. `__main__.py`: The entry point of the application that sets up and runs the FastAPI server.
2. `exposed_module.py`: Contains the custom modules that are exposed through the API.

## Features

- Create and manage sessions
- Execute Horsona modules remotely
- Support for various argument types, including nested Horsona objects
- Expose custom modules through the API

## Usage

To run the Node Graph Server:

1. Clone the repo and navigate to the `node_graph_server/bin/windows` folder. (On Linux: `node_graph_server/bin/linux`)

   ```powershell
   git clone https://github.com/synthbot-anon/horsona.git
   ```

2. Set up the runtime folder.
   ```powershell
   mkdir node_graph_server_runtime
   cp horsona/samples/node_graph_server/bin/windows/node_graph_server.bat node_graph_server_runtime/
   ```
   Then add `.env`, `llm_config.json` and `index_config.json` to the `node_graph_server_runtime` folder. Check the Horsona [README](https://github.com/synthbot-anon/horsona/blob/main/README.md) for more information on how to create these configuration files.

2. Run `node_graph_server.bat`. (On Linux: `node_graph_server.sh`)
   ```powershell
   cd node_graph_server_runtime
   node_graph_server.bat
   ```

   This will pull the latest Docker image and start the server.

3. The server will start running on `http://localhost:8000`.

To update the Docker image to the latest version:

1. Navigate to the `node_graph_server/bin/windows` folder. (On Linux: `node_graph_server/bin/linux`)
2. Run `update - node_graph_server.bat`. (On Linux: `update - node_graph_server.sh`)

To uninstall the Docker image:

1. Navigate to the `node_graph_server/bin/windows` folder. (On Linux: `node_graph_server/bin/linux`)
2. Run `uninstall - node_graph_server.bat`. (On Linux: `uninstall - node_graph_server.sh`)

## Example: Running a custom function

The sample code exposes a `hello_world` function through `node_graph_server.exposed_module`. Here's an example of how to use curl commands to run it:

1. Create a new session and get the session ID:

   ```bash
   SESSION_ID=$(curl -X POST http://localhost:8000/api/sessions | jq -r '.session_id')
   ```

2. Run the `hello_world` function:

   ```bash
   curl -X 'POST' \
     "http://localhost:8000/api/sessions/$SESSION_ID/resources/node_graph_server.exposed_module/hello_world" \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{}'
   ```

   This should return a response containing the result "Hello, world!".


3. Delete the session:

   ```bash
   curl -X 'DELETE' \
     "http://localhost:8000/api/sessions/$SESSION_ID" \
     -H 'accept: application/json'
   ```

   This will delete the session and all its associated resources.
