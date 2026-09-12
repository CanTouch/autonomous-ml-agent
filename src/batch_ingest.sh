#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

URL_FILE="${1:-urls.txt}"

if [ ! -f "$URL_FILE" ]; then
    echo "Error: File '$URL_FILE' not found." >&2
    exit 1
fi

echo "=================================================="
echo "Batch Ingesting YouTube URLs from: $URL_FILE"
echo "=================================================="

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

TOTAL=0
SUCCESS=0
FAILED=0

while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading/trailing whitespace
    url="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    
    # Skip empty lines and comment lines
    if [ -z "$url" ] || [[ "$url" == \#* ]]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] Processing: $url"
    
    if "$PYTHON_BIN" "$PROJECT_ROOT/src/ingest_youtube.py" "$url"; then
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[!] Failed to ingest: $url" >&2
        echo "$(date '+%Y-%m-%d %H:%M:%S') | $url | Subtitles unavailable or disabled" >> "$PROJECT_ROOT/failed_ingests.log"
        FAILED=$((FAILED + 1))
    fi
done < "$URL_FILE"

echo ""
echo "=================================================="
echo "Batch Ingestion Finished!"
echo "Total: $TOTAL | Succeeded: $SUCCESS | Failed: $FAILED"
echo "=================================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
