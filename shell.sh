#!/bin/bash

projroot=$(dirname "$(readlink -f $0)")
poetry shell -C "$projroot"

