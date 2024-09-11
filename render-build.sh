#!/usr/bin/env bash

# Install graphviz
apt-get update && apt-get install -y graphviz

# Continue with the default build process
pip install -r requirements.txt
