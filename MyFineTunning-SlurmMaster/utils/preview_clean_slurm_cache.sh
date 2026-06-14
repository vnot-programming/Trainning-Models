#!/usr/bin/env bash
# Preview script for Slurm Master cache cleanup (dry-run)
# This script lists files/directories that WOULD be removed by the actual cleanup.

set -euo pipefail

HOME_DIR="${HOME}"

echo "=== PREVIEW CLEANUP START $(date) ==="

echo "--- __pycache__ directories ---"
find "${HOME_DIR}" -type d -name '__pycache__' -print

echo "\n--- Ollama cache (files) ---"
find "${HOME_DIR}/.ollama/cache" -type f -print || echo "(Ollama cache directory not present)"

echo "\n--- VSCode JSON-schema cache ---"
find "${HOME_DIR}/.antigravity-ide-server/data/User/globalStorage/vscode.json-language-features/json-schema-cache" -type f -print || echo "(VSCode schema cache not present)"

echo "\n--- Antigravity internal caches (excluding GEMINI.md) ---"
find "${HOME_DIR}/.gemini/antigravity" -type f -print || echo "(Antigravity internal cache dir not present)"
find "${HOME_DIR}/.gemini/config" -type f ! -name 'GEMINI.md' -print || echo "(Gemini config dir not present)"

echo "\n--- Large log files (>10M) in .antigravity-server (excluding *.token) ---"
find "${HOME_DIR}/.antigravity-server" -type f -name '*.log' -size +10M -print || echo "(No large log files)"

echo "\n--- Generic .cache files (excluding torch hub checkpoints) ---"
find "${HOME_DIR}/.cache" -mindepth 1 -maxdepth 1 ! -name 'torch' -print || echo "(No generic .cache files)"

echo "\n--- AI completion cache files (excluding config.json) ---"
find "${HOME_DIR}/.ai_completion" -mindepth 1 -maxdepth 1 ! -name 'config.json' -print || echo "(No AI completion cache)"

echo "=== PREVIEW CLEANUP END $(date) ==="
