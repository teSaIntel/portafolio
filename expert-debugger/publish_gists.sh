#!/usr/bin/env bash
# Creates one public gist per folder (README + code + notebook + figures) with the GitHub CLI.
# Run it from a shell logged in to your *personal* account: gh auth status
set -euo pipefail
cd "$(dirname "$0")"
for d in 00-architecture 01-failure-ranker 02-test-issue-detector 03-knowledge-layer 04-bug-matcher 05-rca-agent; do
  desc="Expert Debugger · $(head -1 "$d/README.md" | sed 's/^# //')"
  gh gist create --public -d "$desc" "$d"/README.md "$d"/*.py "$d"/*.ipynb
done
echo "Gists don't render PNGs: push the folder to a repo for the figures, or drag them into each gist from the web UI."
