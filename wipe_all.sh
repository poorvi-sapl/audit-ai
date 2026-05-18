#!/bin/bash
# ============================================================
# AuditAI — Full data wipe
# Clears Qdrant collections + all local data files
# Run from project root: bash wipe_all.sh
# ============================================================

set -e

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

echo "=== AuditAI Full Data Wipe ==="
echo "Qdrant: ${QDRANT_HOST}:${QDRANT_PORT}"
echo ""

# ── 1. Qdrant collections ────────────────────────────────────────────────────
echo "[ 1/4 ] Deleting Qdrant collections..."

curl -s -X DELETE \
  "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/auditai_sop" \
  -H "Content-Type: application/json" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('  auditai_sop:', r.get('result') or r.get('status') or r)"

curl -s -X DELETE \
  "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/auditai_workpaper_chunks" \
  -H "Content-Type: application/json" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('  auditai_workpaper_chunks:', r.get('result') or r.get('status') or r)"

echo ""

# ── 2. Review queue ──────────────────────────────────────────────────────────
echo "[ 2/4 ] Clearing review queue..."
> data/review_queue.jsonl
echo "  data/review_queue.jsonl — cleared"

# ── 3. JSONL training outputs ────────────────────────────────────────────────
echo "[ 3/4 ] Clearing training pair outputs..."

for f in data/stage2_domain.jsonl data/stage3_firm.jsonl; do
  if [ -f "$f" ]; then
    > "$f"
    echo "  $f — cleared"
  else
    echo "  $f — not found (skipped)"
  fi
done

# ── 4. Suggested aliases CSV ─────────────────────────────────────────────────
echo "[ 4/4 ] Clearing suggested aliases..."
if [ -f "data/suggested_aliases.csv" ]; then
  # Preserve the header row only
  head -1 data/suggested_aliases.csv > data/suggested_aliases.csv.tmp \
    && mv data/suggested_aliases.csv.tmp data/suggested_aliases.csv
  echo "  data/suggested_aliases.csv — header preserved, data cleared"
else
  echo "  data/suggested_aliases.csv — not found (skipped)"
fi

echo ""
echo "=== Done. Re-embed your SOPs in Section 0 before processing workpapers. ==="