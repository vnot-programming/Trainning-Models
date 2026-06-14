#!/usr/bin/env bash
# Actual cleanup script for Slurm Master cache (non‑dry‑run)
# Deletes cache directories after safety checks.

set -euo pipefail

HOME_DIR="${HOME}"
LOG_DIR="${HOME}/cleanup_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cleanup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== CLEANUP START $(date) ==="

echo "--- Disk usage before ---"
df -h "${HOME}" | tail -1

# 1. __pycache__
 echo "[1] Removing __pycache__ directories..."
 find "${HOME_DIR}" -type d -name '__pycache__' -exec rm -rf {} +

# 2. Ollama cache
 echo "[2] Removing Ollama cache (models will be re‑downloaded)"
 rm -rf "${HOME_DIR}/.ollama/cache"

# 3. VSCode JSON‑schema cache
 echo "[3] Removing VSCode JSON‑schema cache"
 rm -rf "${HOME_DIR}/.antigravity-ide-server/data/User/globalStorage/vscode.json-language-features/json-schema-cache"

# 4. Antigravity internal caches (keep GEMINI.md)
 echo "[4] Removing Antigravity internal caches (except GEMINI.md)"
 rm -rf "${HOME_DIR}/.gemini/antigravity"
 # Keep GEMINI.md, delete other config files
 find "${HOME_DIR}/.gemini/config" -type f ! -name 'GEMINI.md' -delete || true

# 5. Large log files (>10M) in .antigravity-server (keep *.token)
 echo "[5] Removing large log files (>10M) in .antigravity-server (excluding *.token)"
 find "${HOME_DIR}/.antigravity-server" -type f -name '*.log' -size +10M -exec rm -f {} + || true

# 6. Generic .cache files (excluding torch hub checkpoints)
 echo "[6] Cleaning generic .cache files (preserving torch checkpoints)"
 find "${HOME_DIR}/.cache" -mindepth 1 -maxdepth 1 ! -name 'torch' -exec rm -rf {} + || true

# 7. AI completion cache files (excluding config.json)
 echo "[7] Cleaning AI completion cache (preserving config.json)"
 find "${HOME_DIR}/.ai_completion" -mindepth 1 -maxdepth 1 ! -name 'config.json' -exec rm -rf {} + || true

echo "--- Disk usage after ---"
df -h "${HOME}" | tail -1

echo "=== CLEANUP END $(date) ==="
