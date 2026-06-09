#!/bin/bash
# Script untuk sinkronisasi aman dari branch slurm ke main
# Script ini menjamin folder MyFineTunning-dev dan MyFineTunning-RunPOD tidak akan terhapus di main.

set -e

echo "=========================================================="
echo "🔄 Auto-Sync: slurm -> main (Preserving Folders)"
echo "=========================================================="

# Pindah ke direktori root repository
cd "$(dirname "$0")/../.."

# Pastikan status git bersih
if [ -n "$(git status --porcelain)" ]; then
  echo "❌ Error: Working directory tidak bersih."
  echo "Harap commit atau stash perubahan Anda di branch saat ini sebelum melakukan sync."
  exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)

echo "[1/6] Pindah ke branch main dan pull update terbaru..."
git checkout main
git pull origin main

echo "[2/6] Melakukan merge dari slurm tanpa commit..."
# Mengabaikan error jika sudah up-to-date
git merge slurm --no-commit || echo "Tidak ada konflik fatal, melanjutkan..."

echo "[3/6] Memulihkan folder dev dan RunPOD dari branch dev..."
git checkout dev -- MyFineTunning-dev MyFineTunning-RunPOD || echo "Folder sudah ada, tidak perlu dipulihkan."

echo "[4/6] Menyimpan perubahan..."
# Cek apakah ada yang perlu di-commit
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Auto-merge: update dari slurm, mempertahankan folder dev dan RunPOD"
    echo "[5/6] Push ke origin main..."
    git push origin main
else
    echo "👍 Sudah up-to-date. Tidak ada yang perlu di-commit."
fi

echo "[6/6] Kembali ke branch ${CURRENT_BRANCH}..."
git checkout "${CURRENT_BRANCH}"

echo "✅ Sinkronisasi otomatis selesai dan sukses!"
echo "=========================================================="
