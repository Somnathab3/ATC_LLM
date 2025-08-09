#!/bin/bash
# ATC-LLM Unix/Linux Shell Script
# This script provides easy access to the ATC-LLM CLI on Unix-like systems

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/cli.py" "$@"
