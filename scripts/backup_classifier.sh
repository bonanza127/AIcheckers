#!/usr/bin/env bash
set -euo pipefail

# Backup the DINOv3 classifier with a timestamp for easy rollback.

SOURCE_PATH="${1:-models/dinov3_classifier.pt}"
BACKUP_DIR="${2:-models}"

if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "Classifier not found: $SOURCE_PATH" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

timestamp="$(date +"%Y%m%d_%H%M%S")"
base_name="$(basename "$SOURCE_PATH" .pt)"
backup_path="${BACKUP_DIR}/${base_name}.backup_${timestamp}.pt"

cp "$SOURCE_PATH" "$backup_path"
echo "Backed up to: $backup_path"
