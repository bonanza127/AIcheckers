#!/bin/bash
cd /home/techne/aicheckers

echo "=== Starting AI extraction ==="
python3 -u scripts/batch_extract.py

echo "=== AI extraction complete, starting Real extraction ==="
python3 -u scripts/extract_real.py

echo "=== All extractions complete! ==="
ls -lh embeddings/
