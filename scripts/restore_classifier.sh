#!/usr/bin/env bash
set -euo pipefail

# Restore the DINOv3 classifier from a backup file.

BACKUP_PATH="${1:-}"
DEST_PATH="${2:-models/dinov3_classifier.pt}"

if [[ -z "$BACKUP_PATH" ]]; then
  echo "Usage: $0 <backup_path> [dest_path]" >&2
  exit 1
fi

if [[ ! -f "$BACKUP_PATH" ]]; then
  echo "Backup not found: $BACKUP_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST_PATH")"
cp "$BACKUP_PATH" "$DEST_PATH"
echo "Restored to: $DEST_PATH"
