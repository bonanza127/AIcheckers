#!/bin/bash
# AIcheckers Data Backup Script
# Backs up critical user data to timestamped directory

set -e  # Exit on error

# Configuration
DATA_DIR="/home/techne/aicheckers/data"
BACKUP_BASE_DIR="/home/techne/aicheckers/data_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_BASE_DIR}/backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Backup files
echo "[$(date)] Starting backup to ${BACKUP_DIR}..."

if [ -d "${DATA_DIR}" ]; then
    cp -r "${DATA_DIR}"/* "${BACKUP_DIR}/" 2>/dev/null || true
    echo "[$(date)] Backed up data directory"
else
    echo "[$(date)] Warning: Data directory not found"
fi

# Count backed up files
FILE_COUNT=$(find "${BACKUP_DIR}" -type f | wc -l)
echo "[$(date)] Backup complete: ${FILE_COUNT} files backed up"

# Clean up old backups (keep last 30 days)
find "${BACKUP_BASE_DIR}" -type d -name "backup_*" -mtime +30 -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] Cleaned up old backups (>30 days)"

# Optional: Sync to remote (uncomment if needed)
# rsync -avz "${BACKUP_DIR}" user@remote:/path/to/backups/

echo "[$(date)] Backup completed successfully"
