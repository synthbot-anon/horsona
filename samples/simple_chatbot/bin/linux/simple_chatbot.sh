#!/bin/bash

docker run --rm -it --network host -v "$(pwd)":/host synthbot/simple_chatbot
