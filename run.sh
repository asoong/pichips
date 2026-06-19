#!/usr/bin/env bash
# Start the PiChip viewer on the Pi using its venv. Reads config from .env.
cd "$(dirname "$0")"
exec .venv/bin/python overlay_viewer.py "$@"
