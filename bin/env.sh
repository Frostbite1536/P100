#!/bin/bash
# Sourced by every script. Override per machine.
P100_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export P100_HOME
export P100_RECEIPTS="$P100_HOME/state/receipts"
export P100_WT="$P100_HOME/.worktrees"
export P100_PROFILES="$P100_HOME/profiles"
export P100_PY="$P100_HOME/.venv/bin/python"
# Verify before first use: `codex exec -m <id>` must be a model your CLI accepts.
export WORKER_MODEL="${WORKER_MODEL:-gpt-6.1-sol}"
export WORKER_EFFORT="${WORKER_EFFORT:-high}"
