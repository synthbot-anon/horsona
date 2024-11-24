#!/bin/bash

docker run --rm -it -p 8000:8000 -v "$(pwd)":/host synthbot/node_graph_server
