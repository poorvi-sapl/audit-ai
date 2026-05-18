# ETL & Pipeline Design — Audit AI
**Version:** 1.0  
**Date:** April 2026  
**Status:** FINAL  
**Checklist items closed:** #9 (ETL Flow), #10 (Pipeline Flow), #11 (Data Processing Layers), #15 (Long Context Problem)

---

## 1. Design Principles

1. **Raw content is never modified.** MongoDB `raw_workpapers` stores exactly what was extracted. All cleaning happens downstream in the transformer chain and is recorded in `cleaning_log`.
2. **Every cleaning action is logged.** One row per action per workpaper in `cleaning_log`. If a bug is found later, the lineage exists to diagnose it.
3. **Split at engagement level, never at row or pair level.** All workpapers from one engagement go into one split partition. This prevents data leakage. (Edge case D-08.)
4. **The pipeline never touches draft or active engagements.** The arq poller checks `engagements.status = 'final'` before triggering any ETL work.
5. **All numeric comparisons are done in code, never by the model.** The `model/tools/` math library runs via LangGraph before generation and injects computed facts into the context window. The model explains and documents; it does not calculate. (Edge case I-02.)
6. **Truncation always happens at complete row/sentence boundaries.** Never at token boundaries. (Edge case D-06, I-05.)

---

## 2. Airflow DAG Architecture

### DAG 1 — intake_watcher

**Schedule:** Every 5 minutes (configurable via Airflow Variable `INTAKE_POLL_INTERVAL`)  
**Purpose:** Detect new folders in `/intake/`, validate against PG, trigger ETL.

```
intake_watcher
│
├── sense_new_folders
│     FileSensor polls /intake/ for folders not yet in PG workpapers
│
├── for each new folder:
│   ├── parse_folder_name
│   │     Extract engagement_code from folder name
│   │
│   ├── validate_engagement
│   │     PG lookup: engagements WHERE engagement_code=? AND status='final'
│   │     SKIP if not found or status != 'final'
│   │     SKIP if engagement already being processed (check PG workpapers.extraction_status)
│   │
│   └── trigger_etl_pipeline
│         TriggerDagRunOperator → etl_pipeline DAG
│         conf: {engagement_id, folder_path, file_list}
```

---

### DAG 2 — etl_pipeline

**Triggered by:** intake_watcher (per engagement folder)  
**Purpose:** Extract, transform, load, chunk, embed all workpapers in one engagement folder.

```
etl_pipeline
│
├── dedup_check
│     For each file:
│       Compute SHA-256 hash
│       Acquire Redis lock: key = f"wp_ingest:{file_hash}", ttl = 300s
│         If lock not acquired within 10s → skip file, log WARN
│           ("Another worker is already processing this file — skipping to avoid duplicate")
│       With lock held:
│         Query PG: SELECT id FROM workpapers WHERE file_hash=?
│         If row exists → release lock, skip file (log info: "already processed")
│         Else → hold lock and proceed to create_workpaper_rows
│     Rationale: the SELECT-then-INSERT is not atomic. Without the Redis lock, two
│     concurrent DAG runs for the same engagement can both read "no row found" and
│     both attempt INSERT — the second hits the DB UNIQUE(file_hash) constraint.
│     The lock prevents the race at the application layer; the UNIQUE constraint
│     is the last-resort guard if the lock is bypassed (e.g. Redis outage).
│
├── read_engagement_labels
│     Look for engagement_labels.csv in the intake folder
│     FAIL with WARN (not hard fail) if file missing — flag engagement for manual label entry
│     Parse CSV → dict keyed by workpaper_ref:
│       {workpaper_ref, workpaper_type, audit_phase, time_sensitivity, primary_regulation, notes}
│     Validate enum values (import WorkpaperType from db.enums — single source of truth):
│       workpaper_type ∈ WorkpaperType — 13 canonical values:
│                         bank_reconciliation, trial_balance, financial_statements,
│                         analytical_procedure, sefa_schedule, compliance_test,
│                         internal_control, risk_assessment, finding_documentation,
│                         management_letter, planning_document, other, final_report
│       audit_phase    ∈ {planning, fieldwork, completion, reporting}
│       time_sensitivity ∈ {timeless, time_static, time_sensitive}
│     Any invalid workpaper_type → reject submission (HTTP 422); do NOT default silently
│     Any invalid audit_phase or time_sensitivity → log error, default to time_static
│
├── create_workpaper_rows
│     Parse each filename: prefix→workpaper_ref, body→section, ext→file_type
│     Look up labels dict for this workpaper_ref
│     INSERT INTO workpapers (extraction_status='pending', pii_scrubbed=FALSE,
│                             workpaper_type=?, audit_phase=?, time_sensitivity=?,
│                             primary_regulation=?, file_hash=?)
│       ON CONFLICT (file_hash) DO NOTHING   ← DB constraint catches any race that
│                                              slipped through the Redis lock
│     If conflict detected → log info ("dedup race caught by DB UNIQUE constraint"),
│                             release Redis lock, skip remainder for this file
│     Else → INSERT INTO audit_trail (event_type='created')
│            Release Redis lock
│
├── extract  [per file, parallel with max_active_tasks=4]
│     │
│     ├── check_password_protection  [runs before detect_extractor on every file]
│     │     │
│     │     ├── PDF (.pdf magic bytes 25 50 44 46):
│     │     │     fitz.open(path)
│     │     │     If doc.is_encrypted == True → password_protected = True
│     │     │
│     │     ├── Excel (.xlsx / .xls):
│     │     │     msoffcrypto.OfficeFile(open(path, 'rb')).is_encrypted()
│     │     │     If True → password_protected = True
│     │     │     (msoffcrypto handles both xlsx ECMA encryption and xls legacy RC4)
│     │     │
│     │     ├── ZIP (.zip):
│     │     │     zipfile.ZipFile(path)
│     │     │     Attempt zf.open(zf.namelist()[0]) without password
│     │     │     If RuntimeError('encrypted') → password_protected = True
│     │     │
│     │     └── If password_protected == True:
│     │           UPDATE PG workpapers SET extraction_status='needs_password'
│     │           INSERT cleaning_log(action='password_protected', detail_json={file_type, detected_by})
│     │           Notify engagement manager via email/Slack:
│     │             "Workpaper {workpaper_ref} is encrypted. Supply password via
│     │              PATCH /workpapers/{id}/password to continue extraction."
│     │           Release Celery worker — do NOT proceed to detect_extractor
│     │           Worker picks up next queued file immediately
│     │
│     ├── detect_extractor
│     │     Check magic bytes and extension:
│     │       50 4B 03 04                    → xlsx/xlsm (ZIP-based)
│     │       D0 CF 11 E0                    → xls/doc (binary OLE)
│     │       25 50 44 46                    → pdf
│     │       3C 3F 78 6D or EF BB BF 3C    → XML — then sniff root element:
│     │           root == ENVELOPE or TALLYMESSAGE → TallyXMLExtractor
│     │           other XML root             → flag for manual review (unknown XML)
│     │       .tdl extension                 → WordExtractor (plain text path; no numeric chunking)
│     │       .dbf / .fpt / .bak / .cgf     → DBFExtractor
│     │     Select extractor from registry
│     │
│     ├── run_extractor
│     │     Call extractor.extract() → RawContent
│     │     On ExtractionError: try fallback extractor
│     │     On all fallbacks fail: set extraction_status='error', INSERT cleaning_log(action='incomplete_removed')
│     │
│     ├── store_raw
│     │     INSERT MongoDB raw_workpapers (full extracted content, no cleaning)
│     │     UPDATE PG workpapers SET mongo_raw_id=?, extraction_status='extracted'
│     │
│     └── emit_transform_task (XCom: workpaper_id)
│
├── transform  [per workpaper, sequential chain]
│     │
│     ├── normalizer
│     ├── pii_scrubber
│     ├── temporal_tagger
│     ├── chunker
│     └── injection_sanitizer
│     (each transformer appends to record.cleaning_actions)
│     (INSERT cleaning_log rows after full chain completes)
│     UPDATE PG workpapers SET pii_scrubbed=TRUE
│
├── load_chunks
│     INSERT PG workpaper_chunks (content_json JSONB, embedding_synced=FALSE)
│     UPDATE PG workpapers SET extraction_status='chunked', total_tokens=SUM(token_count)
│
├── embed_chunks  [per chunk, batched in groups of 32]
│     e5-mistral-7b-instruct.encode(chunk.content_json.text)
│     Qdrant upsert → workpaper_chunks_embeddings (point_id=chunk.id)
│       payload: {workpaper_id, engagement_id, workpaper_type, year,
│                 chunk_index, token_count, chunk_mode, is_rollforward,
│                 source_engagement_id, drift_weight}   ← drift_weight from TemporalTagger
│     UPDATE PG workpaper_chunks SET embedding_synced=TRUE
│
├── check_labels_complete                 ← HARD GATE before pair building
│     SELECT COUNT(*) FROM workpapers
│       WHERE engagement_id = :eid
│         AND workpaper_type IS NULL
│     If count > 0:
│       raise AirflowFailException(
│           f"{count} workpaper(s) in engagement {eid} have no workpaper_type label. "
│           "Add a row for every workpaper to engagement_labels.csv and re-trigger."
│       )
│     Rationale: build_sft_pairs uses workpaper_type to select the prompt template and
│     task_type for every training pair. A null workpaper_type would silently produce
│     pairs with a wrong or missing task_type, corrupting the training set.
│     Hard-failing here is safer than filtering — it forces the labels to be completed
│     before any pair is written, rather than silently skipping unlabelled workpapers.
│
├── build_sft_pairs
│     SFT pair builder (see Section 8)
│     INSERT MongoDB training_pairs_content
│     INSERT PG sft_training_pairs (reviewer_approved=FALSE)
│
└── finalize
      UPDATE PG workpapers SET status='ready'
      INSERT audit_trail (event_type='status_changed', metadata={extraction_status:'chunked'})
      UPDATE PG engagements SET total_workpapers = total_workpapers + N
```

**Password-protected file retry flow** (`PATCH /workpapers/{id}/password`):

```
Engagement manager receives notification → supplies password via API

FastAPI PATCH /workpapers/{id}/password
    │
    ├── Validate: extraction_status must == 'needs_password' (HTTP 400 otherwise)
    ├── Validate: caller has write permission (workpaper_permissions check)
    │
    ├── Decrypt in-memory (never write decrypted file to disk):
    │     PDF:   pikepdf.open(path, password=supplied_password) → BytesIO
    │     Excel: msoffcrypto.OfficeFile.decrypt(supplied_password) → BytesIO
    │     ZIP:   zipfile.ZipFile.open(member, pwd=supplied_password.encode()) → BytesIO
    │     On DecryptionError (wrong password): return HTTP 400 {"detail": "Incorrect password"}
    │
    ├── Write decrypted bytes to a temp path (mkstemp, mode=0o600)
    │     Path is in-process only — OS temp dir, never the intake folder
    │
    ├── Re-queue ETL Celery task with temp_path override
    │     Task cleans up the temp file after extraction completes
    │
    ├── UPDATE PG workpapers SET extraction_status='pending'
    │     (resets to pending; DAG 2 resumes from extract step)
    │
    └── Return HTTP 202 Accepted {"detail": "Re-queued for extraction"}
    NOTE: password is never logged, never written to DB, never stored anywhere
```

---

### DAG 3 — jsonl_export

**Triggered by:** Manual — `role = 'admin'` required to trigger via Airflow REST API (`POST /api/v1/dags/jsonl_export/dagRuns`). Airflow RBAC enforces this at the trigger endpoint; non-admin accounts receive HTTP 403. Only the ML Engineer (admin role) should trigger this DAG.  
**Parameters:** `--mode [mass|batch]`, `--framework [single_audit|gagas|gaas]` (batch mode only)  
**Purpose:** Export approved SFT pairs to JSONL files for training. Runs once per stage. Output files contain client workpaper text — treat as confidential.

```
jsonl_export
│
├── resolve_export_scope
│     mass mode:  all engagements (Single Audit + GAGAS + GAAS — US frameworks only)
│     batch mode: filtered by --has_single_audit or --client_type param
│
├── fetch_approved_pairs
│     SELECT sft_training_pairs WHERE reviewer_approved=TRUE
│       AND client_type IN (resolved_scope)
│     Group by split_assignment (train / val / test)
│     Verify engagement-level integrity: no engagement spans two splits
│     NOTE: would_sign_off rate is NOT a gate here — it gates inference expansion
│           at the API layer, not training data export
│
├── verify_task_balance
│     Count pairs per pair_type
│     WARN if any single pair_type exceeds 40% of total pairs
│
├── fetch_content  [per pair]
│     MongoDB training_pairs_content lookup by mongo_pair_id
│
├── write_jsonl  [three files, output path includes mode/framework tag]
│     mass mode:
│       data/training/mass/train.jsonl  (70% — all frameworks mixed)
│       data/training/mass/val.jsonl    (15%)
│       data/training/mass/test.jsonl   (15% — holdout)
│     batch mode (e.g. single_audit):
│       data/training/single_audit/train.jsonl
│       data/training/single_audit/val.jsonl
│       data/training/single_audit/test.jsonl
│
├── secure_output_files
│     os.chmod on all written .jsonl files → 0o600 (owner read/write only)
│     os.chown → train_svc:train_svc (the OS service account that runs training_trigger)
│     Verify: ETL service account (etl_svc) cannot read these files — assert PermissionError
│     The inference service account (inf_svc) also has no access to data/training/
│     INSERT audit_trail (event_type='jsonl_exported', metadata={file_count, mode, framework, pair_count})
│
├── hash_dataset
│     SHA-256 of train.jsonl → stored in training_runs.dataset_hash for reproducibility
│
└── version_with_dvc
      dvc add data/training/{mode_or_framework}/
      Git commit version tag
      Write .pending_deletion marker: data/training/{mode_or_framework}/.pending_deletion
        → training_trigger DAG reads this marker and deletes JSONL files after run_sft completes
        → JSONL files exist on disk only during the window between export and training completion
```

---

### DAG 4 — training_trigger

**Triggered by:** Manual (after JSONL export for each stage)  
**Parameters:** `--stage [mass|batch|refine]`, `--framework [single_audit|gagas|gaas]` (batch/refine only), `--base_checkpoint` (path to prior stage checkpoint)  
**Purpose:** Run SFT fine-tuning for the specified stage, evaluate, and register new model version. Called once per stage/track.

```
training_trigger
│
├── pre_flight_checks  (maps to Pre-Training Checklist — Section 10)
│     Verify all 20 checklist items programmatically
│     FAIL immediately if any CRITICAL item fails
│
├── resolve_dataset_path
│     mass stage:  data/training/mass/
│     batch stage: data/training/{framework}/
│     refine stage: data/training/{framework}_curated/
│
├── resolve_base_checkpoint
│     mass stage:  base Mistral 22B NF4 weights (no prior adapter)
│     batch stage: /models/mass-v{n}/  ← always continues from Stage 1, never from base
│     refine stage: /models/{framework}-v{n}/
│
├── insert_training_run
│     INSERT PG training_runs (status='queued', run_type=stage, framework=?, dataset_hash, pair_count, epochs)
│
├── run_sft
│     python model/training/train.py
│       --dataset_path resolved_dataset_path
│       --base_checkpoint resolved_base_checkpoint
│       --run_type {stage}
│     mass stage:  LR=2e-4, epochs=3, batch=4
│     batch stage: LR=1e-4 (lower — fine-tuning atop mass checkpoint), epochs=2, batch=4
│     refine stage: LR=5e-5 (conservative), epochs=1, batch=4
│     Checkpoints every 500 steps → /models/{run_id}/checkpoints/
│     Validate each checkpoint after save (N-01 fix)
│     MLflow log: loss curves, LR schedule, GPU utilization, stage tag
│     UPDATE PG training_runs SET status='running'
│
├── post_training_cleanup
│     Check for .pending_deletion marker in resolved_dataset_path
│     If present:
│       Delete all .jsonl files in data/training/{mode_or_framework}/
│       Delete .pending_deletion marker
│       INSERT audit_trail (event_type='jsonl_deleted', metadata={mode, framework, deleted_at})
│       Log: "JSONL files deleted after training — client data not retained on disk"
│     If absent (marker missing): log WARNING, do not delete (manual investigation needed)
│     JSONL files are retained only during the ETL→training window — not persisted beyond
│
├── run_eval
│     python model/training/eval.py --split test --dataset resolved_dataset_path
│     mass stage:  F1 per task × per framework — all four frameworks scored
│     batch stage: F1 per task for this framework only + regression on mass holdout
│     refine stage: Full eval suite + prior framework holdout (regression check)
│     UPDATE PG training_runs SET f1_after=?, hallucination_after=?, status='completed'
│
├── register_version
│     INSERT PG model_versions (version_tag='{stage}-{framework}-v{n}', adapter_path, f1_score, hallucination_rate)
│     is_current = FALSE until partner signs off
│     single_audit batch adapter → promoted to is_current for Phase 1 inference after sign-off
│
└── notify_partner
      Alert partner tier users: "New model version ready for sign-off"
      Include: stage, framework, F1 delta vs prior version
```

---

### DAG 5 — dpo_training

**Triggered by:** Manual (quarterly, after DPO pair threshold met)  
**Purpose:** Run DPO alignment cycle on accumulated feedback corrections.

```
dpo_training
│
├── validate_dpo_gate
│     SELECT COUNT(*) FROM dpo_candidates WHERE used_in_run=FALSE
│     FAIL if count < 300 (P-05 fix)
│
├── filter_pairs  (P-01, P-02 fixes)
│     Keep only reviewer_tier IN ('partner','senior')
│     Remove pairs where similarity(chosen, rejected) > 0.95
│     Log count of filtered pairs
│
├── fetch_pair_content
│     MongoDB training_pairs_content WHERE pair_category='dpo'
│
├── insert_training_run
│     INSERT PG training_runs (run_type='dpo', base_model_version=current_is_current)
│
├── run_dpo
│     python model/training/dpo.py
│     beta=0.1, max_epochs=2 (P-03 fix)
│     MLflow log all metrics
│
├── run_eval
│     Full eval suite + general capability test set (P-03 fix)
│     Compare F1 before/after
│
├── mark_pairs_used
│     UPDATE PG dpo_candidates SET used_in_run=TRUE, training_run_id=?
│
├── register_version
│     INSERT PG model_versions
│
└── notify_partner
      Alert for sign-off before activation
```

---

## 3. Intake Folder Parser

```python
# airflow/dags/intake_watcher.py (key logic)

import re
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r'^(?P<ref>[A-Z0-9\-]+)_(?P<section>.+)\.(?P<ext>[a-zA-Z0-9]+)$'
)

def parse_intake_folder(folder_path: Path) -> dict:
    engagement_code = folder_path.name          # "SA-2022-001"
    files = []
    for f in folder_path.iterdir():
        if not f.is_file():
            continue
        m = FILENAME_PATTERN.match(f.name)
        if not m:
            # Log unparseable filename — queue for manual review
            continue
        ext = m.group('ext').lower()
        if ext in ('cdx','cvw','sty','mdx','ntx'):
            continue                             # always skip — no data
        files.append({
            'path':          str(f),
            'workpaper_ref': m.group('ref'),     # "C-1"
            'section':       m.group('section'), # "CDBG_Compliance"
            'extension':     ext,
        })
    return {'engagement_code': engagement_code, 'files': files}
```

---

## 4. Per-Format Extraction Pipelines

### 4.1 Excel (.xlsx / .xls / .xlsm)

```
Receive file
    │
    ├── Verify magic bytes
    │     50 4B 03 04 → xlsx (ZIP-based)
    │     D0 CF 11 E0 → xls (binary)
    │
    ├── PRIMARY: pandas + openpyxl
    │     xl = pd.ExcelFile(path, engine='openpyxl')
    │     Skip sheet names matching: Cover|Index|TOC|Instructions|Summary (regex)
    │     For each valid sheet:
    │       df = xl.parse(sheet, dtype=str)            ← D-05 fix: always dtype=str
    │       wb = openpyxl.load_workbook(path, data_only=True)  ← D-03 fix
    │       Handle merged cells → forward-fill (D-02 fix)
    │       df.fillna('')                              ← D-01 fix
    │       Detect subtotals (rows = sum of preceding rows) → mark subtotal=True
    │       Drop rows where >85% cells are empty
    │       Prepend sheet_name column to every row
    │
    ├── FALLBACK 1 (if .xls or openpyxl magic byte fail):
    │     xlrd.open_workbook(path)
    │     Convert to xlsx via LibreOffice headless
    │     Retry pandas path above
    │
    ├── FALLBACK 2 (if .xlsm / macro-enabled):
    │     xlwings.Book(path, read_only=True)
    │     Extract values only — skip VBA
    │
    ├── FALLBACK 3 (all pandas attempts fail):
    │     LibreOffice headless → CSV
    │     pd.read_csv() — loses sheet names, logs warning
    │
    └── LAST RESORT: flag for manual queue
          INSERT cleaning_log(action='incomplete_removed')
          SET workpapers.extraction_status='error'
```

**Validation after extraction (pydantic + great_expectations):**
- Column type consistency check
- Null rate assertion (flag if >60% of rows in any column are empty)
- Numeric value range assertions for monetary columns

---

### 4.2 Word (.docx / .doc)

```
Receive file
    │
    ├── PRIMARY: python-docx
    │     doc = Document(path)
    │     Extract paragraphs with style metadata (Heading1/2, Normal, etc.)
    │     Extract tables → [{headers, rows}]
    │     Detect and skip: attorney letters, board minutes
    │       (heuristic: paragraphs starting with "PRIVILEGED" or "ATTORNEY-CLIENT")
    │     Detect track changes → if present, accept all changes via LibreOffice first
    │
    ├── FALLBACK (.doc legacy binary):
    │     LibreOffice headless convert .doc → .docx
    │     Retry python-docx path
    │
    └── LAST RESORT: flag for manual queue
```

---

### 4.3 PDF — Text (Digital Native)

```
Receive file
    │
    ├── PRE-FLIGHT: pypdf / pypdf2
    │     Get page_count, rotation, metadata
    │     Detect if PDF is actually scanned (avg chars/page < 100 → route to 4.4 scanned path)
    │     If page_count > 100:
    │       Split into 25-page windows using pypdf PdfReader page slices
    │       Process each window independently → merge results
    │       Reason: pdfplumber can hang on large complex PDFs; 25-page limit bounds worst-case
    │
    ├── TIMEOUT WRAPPER (signal.alarm — Unix only; thread-based on Windows)
    │     Per-window timeout: 90 seconds
    │     On TimeoutError:
    │       Log extraction_timeout (workpaper_id, page_window, elapsed_ms)
    │       Mark window as error_window — continue with remaining windows
    │       If ALL windows timeout: escalate to LAST RESORT
    │
    ├── PRIMARY: pdfplumber (per window)
    │     For each page in window:
    │       page.extract_text() → text blocks
    │       page.extract_tables() → structured tables
    │       Preserve column layout (pdfplumber maintains coordinates)
    │     Combine pages with page_number metadata
    │
    ├── FALLBACK (pdfplumber returns empty for a window):
    │     pdfminer.six LAParams extraction for that window
    │     Lower-level — catches complex column layouts
    │
    └── LAST RESORT: flag for manual queue
          Set extraction_status = 'error'
          INSERT cleaning_log(action='incomplete_removed', detail='pdf_timeout')
          Notify engagement manager: "Workpaper {ref} requires manual extraction — PDF too complex"
          Release Celery worker (worker is not blocked)
```

---

### 4.4 PDF — Scanned (OCR Path)

```
Receive file
    │
    ├── PRE-FLIGHT: page count check
    │     If page_count > 50:
    │       Split into 25-page windows (pypdf PdfReader page slices → temp PDF per window)
    │       Process each window independently through the full OCR pipeline below
    │       Merge results preserving page_number offsets
    │       Reason: Docling/Surya on a 100-page scanned PDF can exceed 30 minutes;
    │               25-page windows cap worst-case OCR time to ~8 minutes per window
    │
    ├── TIMEOUT WRAPPER (per window)
    │     Per-window timeout: 600 seconds (10 min — OCR is slow; scanned pages need time)
    │     On TimeoutError:
    │       Log ocr_timeout (workpaper_id, page_window, elapsed_ms)
    │       Route timed-out window to Tesseract fallback with 120s timeout
    │       If Tesseract also times out: mark window as error_pages[]
    │
    ├── pdf2image (per window)
    │     Convert pages to 300 DPI PNG images — Poppler backend
    │
    ├── OpenCV pre-processing (per page image)
    │     Deskew (correct rotation up to ±5°)
    │     Denoise (fastNlMeansDenoising)
    │     Binarize (adaptive threshold)
    │     → 15-20% OCR accuracy improvement from pre-processing alone
    │
    ├── PRIMARY: Docling + Surya
    │     On-premise, GPU-accelerated on L40S
    │     Returns structured blocks: text, tables, figures
    │     Surya handles multi-column layouts and Hindi text
    │     Confidence score per block (float 0.0–1.0)
    │     If avg confidence < 0.60: route to fallback
    │
    ├── FALLBACK (Surya confidence < 0.60 OR window timed out):
    │     Tesseract 5.3+
    │     lang='eng' (US-only firm; Hindi OCR out of scope)
    │     Confidence threshold: 40% minimum (below → flag as error_page)
    │     Timeout: 120 seconds per window
    │
    └── LAST RESORT: flag for manual queue
          Record error_pages[] in extraction_meta (only timed-out/failed windows)
          Partial extraction is valid: successfully extracted windows are kept
          Set ocr_used=TRUE on workpaper row
          If > 50% of pages are error_pages: set extraction_status='error', notify engagement manager
```

---

### 4.5 DBF / FoxPro

```
Receive file
    │
    ├── Detect format by extension:
    │     .dbf / .fpt → dbfread
    │     .bak / .cgf → chardet detect encoding → configparser or json parse
    │
    ├── PRIMARY: dbfread
    │     dbfread.DBF(path, encoding=detected_encoding)
    │     Returns records as list of dicts
    │     Apply field_mappings/busy.yaml or default.yaml based on column header match
    │
    ├── FALLBACK: simpledbf
    │     simpledbf.Dbf5(path)
    │     Convert to pandas DataFrame
    │
    └── LAST RESORT: flag for manual queue
```

**Skip list (always):** `.cdx`, `.cvw`, `.sty`, `.mdx`, `.ntx` — these are index/structure files with no audit data content.

---

### 4.6 Tally ERP XML (.xml with Tally root element)

Tally ERP exports trial balance, ledger, and voucher data as XML. No new dependency — uses Python stdlib `xml.etree.ElementTree`.

**Supported export types:**

| Tally export | Root element | Mapped to |
|---|---|---|
| Ledger / Trial Balance | `<ENVELOPE><BODY><TALLYMESSAGE><LEDGER>` | `trial_balance` workpaper_type |
| Voucher / Bank transactions | `<ENVELOPE><BODY><TALLYMESSAGE><VOUCHER>` | `bank_reconciliation` workpaper_type |
| Stock / Inventory | `<STOCKITEM>` | `substantive_test` workpaper_type (semantic chunker) |

```
Receive .xml file
    │
    ├── Verify root element
    │     tree = ET.parse(path)
    │     root = tree.getroot()
    │     If root.tag not in ('ENVELOPE', 'TALLYMESSAGE') → flag unknown_xml, manual queue
    │
    ├── Detect export type
    │     Sniff first child of TALLYMESSAGE:
    │       LEDGER elements present  → ledger_mode
    │       VOUCHER elements present → voucher_mode
    │       STOCKITEM elements       → stock_mode (semantic path)
    │
    ├── ledger_mode → produces {label, amount, currency, row_type} rows
    │     For each <LEDGER NAME="...">:
    │       label       = LEDGER[@NAME]
    │       opening     = <OPENINGBALANCE> text → float (strip Dr/Cr suffix)
    │       closing     = <CLOSINGBALANCE> text → float
    │       row_type    = "opening" for opening row, "closing" for closing row
    │       currency    = detect from <CURRENCYNAME> or default to engagement.currency
    │     Emit one row per balance type per ledger account
    │     → content_json.chunk_mode = "numeric"
    │
    ├── voucher_mode → produces {label, amount, currency, row_type} rows
    │     For each <VOUCHER VCHTYPE="..." DATE="...">:
    │       label       = "{VCHTYPE} — {DATE}" (e.g. "Payment — 2024-04-01")
    │       amount      = <ALLLEDGERENTRIES><AMOUNT> → float (negative = credit)
    │       row_type    = "addition" if amount > 0, "deduction" if amount < 0
    │       currency    = detect from <CURRENCYNAME> or default to engagement.currency
    │     → content_json.chunk_mode = "numeric"
    │
    ├── stock_mode → semantic path
    │     Serialize STOCKITEM elements to plain text table (label: value pairs)
    │     → content_json.chunk_mode = "semantic"
    │
    ├── Normalizer applied after extraction (same as all paths)
    │     Strip Dr / Cr suffix from amounts → signed float
    │     Lakh / Crore expansion: 2.5L → 250000, 1.2Cr → 12000000
    │
    └── FALLBACK (ET.parse fails — malformed XML or encoding error):
          Try lxml parser with recovery=True
          If still fails: flag for manual queue, set extraction_status='error'
```

**Edge cases:**
- Tally sometimes exports amounts with `Dr` / `Cr` suffixes instead of sign — normalizer handles this
- Multi-currency Tally files: the `<CURRENCYNAME>` tag is per-ledger; each row stores its own currency
- `.tdl` (Tally Data Language) files: not XML — route through `WordExtractor` (plain text), semantic chunker only

---

## 5. Transformer Chain

Order is fixed. Each transformer receives a `Record` and returns a modified `Record`. Every action is recorded in `record.cleaning_actions` and written to `cleaning_log` after the full chain completes.

### Step 1 — Normalizer

**Purpose:** Standardize formats so downstream transformers and the model see consistent data.

| Operation | Detail |
|---|---|
| Currency normalization | Strip `$`, `,`. Convert `(45,000)` → `-45000`. |
| Date normalization | Parse all date formats → ISO 8601 (YYYY-MM-DD). Handle MM/DD/YYYY (US standard). |
| Whitespace normalization | Collapse multiple spaces/tabs/newlines → single space. Strip leading/trailing. |
| Column header normalization | Lowercase, strip special chars, replace spaces with underscores. |
| Numeric string extraction | Extract numeric value from mixed strings like `$750K`, `$1,234,567.00`. Store both raw string and extracted float. |

---

### Step 2 — PIIScrubber

**Purpose:** Remove all client-identifying information before content can be used for training.

Runs Presidio `AnalyzerEngine` + `AnonymizerEngine` with custom recognizers layered on top.

**US Patterns (pii_patterns.yaml — US section):**

| PII Type | Pattern | Replacement |
|---|---|---|
| EIN | `\b\d{2}-\d{7}\b` | `[EIN]` |
| SSN | `\b\d{3}-\d{2}-\d{4}\b` | `[SSN]` |
| ITIN | `\b9\d{2}-\d{2}-\d{4}\b` | `[ITIN]` |
| Client entity name | Presidio ORG recognizer + `\b[A-Z][a-z]+ (Inc\.|LLC|Corp\.|County|City of)\b` | `[CLIENT_ENTITY]` |
| Account number | `\b\d{8,17}\b` in banking context | `[ACCOUNT_NUM]` |
| Routing number | `\b\d{9}\b` in banking context | `[ROUTING_NUM]` |

*(India PII patterns removed — HCLLP is US-only; no Indian clients.)*

**cleaning_log entries written:**
- `action='pii_removed'`, `pii_types_found=['CLIENT_NAME','EIN']`, `detail_json={locations:[...]}`
- One row per workpaper per PII scrub pass

**Duplicate & rollforward detection (runs in this step):**

```
For each workpaper in the current batch:
  Compute MinHash signature of full_text
  Query cleaning_log for prior workpapers across ALL engagements (not just current)
  If similarity_score > 0.85 AND prior_workpaper is in a different engagement:
    → ROLLFORWARD detected
    INSERT cleaning_log(action='rollforward_detected', is_duplicate_of=prior_id, similarity_score)
    UPDATE workpapers SET source_engagement_id = prior_engagement_id (the engagement being rolled from)
    UPDATE workpaper_chunks SET is_rollforward = TRUE,
                                source_engagement_id = prior_engagement_id
      (applied to ALL chunks of this workpaper after chunking completes)
    Do not add to training pairs (rolled-forward content adds no new signal)
    STILL embed and push to Qdrant — is_rollforward flag excludes from retrieval,
      but chunks remain available for future training data analysis
  If similarity_score > 0.85 AND prior_workpaper is in the SAME engagement:
    → WITHIN-ENGAGEMENT DUPLICATE (re-upload or version)
    INSERT cleaning_log(action='duplicate_flagged')
    Skip entirely — do not embed, do not add to training pairs
  If exact file_hash match (any engagement):
    INSERT cleaning_log(action='duplicate_flagged')
    Skip entirely
```

---

### Step 3 — TemporalTagger

**Purpose:** Tag every training pair with its fiscal year and apply content-type-aware temporal weighting so the model learns from the right data at the right strength — not all old data is equally outdated.

**Year extraction:**
- From `workpapers.year_of_workpaper` (set from filename/engagement fiscal_year)
- Fallback: detect 4-digit years in text, take most frequent

**Two-factor weighting (both factors applied at JSONL export, not at extraction time):**

#### Factor 1 — Year Weight (decay by age)

| Data year | Year weight | Rationale |
|---|---|---|
| 2024–2026 | 1.00 | Current standards — full weight |
| 2022–2023 | 0.85 | Minor threshold changes |
| 2020–2021 | 0.70 | Some guidance updates post-COVID |
| 2018–2019 | 0.50 | Pre-2020 guidance |
| 2015–2017 | 0.30 | Significant regulation changes since |
| Before 2015 | 0.10 | Domain language only — not judgment |

#### Factor 2 — Sensitivity Multiplier (set by audit team in engagement_labels.csv)

| `time_sensitivity` label | Multiplier | Meaning | Example workpaper |
|---|---|---|---|
| `timeless` | Override → **1.0** (year weight ignored) | Fundamental audit methodology that does not expire | How to document a bank reconciliation test; audit objective statements; independence documentation structure |
| `time_static` | **1.0 × year_weight** | Content accurate at the time; may have minor updates but structure is valid | Standard compliance test procedures; sampling methodology workpapers; internal control narratives |
| `time_sensitive` | **0.5 × year_weight** | Directly references thresholds, dollar amounts, or regulation versions that change | SEFA schedules with $ amounts; compliance tests citing specific 2 CFR 200 thresholds; materiality calculations |

#### Combined formula

```python
def compute_final_weight(year: int, time_sensitivity: str) -> float:
    year_weights = {
        range(2024, 2027): 1.00,
        range(2022, 2024): 0.85,
        range(2020, 2022): 0.70,
        range(2018, 2020): 0.50,
        range(2015, 2018): 0.30,
    }
    year_weight = next(
        (w for r, w in year_weights.items() if year in r), 0.10
    )
    if time_sensitivity == 'timeless':
        return 1.0                          # age is irrelevant for methodology
    elif time_sensitivity == 'time_sensitive':
        return round(year_weight * 0.5, 2)  # double decay for threshold-heavy content
    else:                                   # time_static
        return year_weight
```

**Why this matters:** A 2012 workpaper documenting *how* HCLLP conducts a bank reconciliation test is just as valuable as a 2024 one — the methodology hasn't changed (`timeless` → weight 1.0). A 2012 SEFA schedule referencing the pre-2014 OMB A-133 $500K threshold is actively harmful if the model learns from it at full weight (`time_sensitive` → weight 0.05 = 0.10 × 0.5).

**outdated_regulations.json:** Maps regulation IDs to their supersession date. If a training pair cites a regulation that was superseded before the fiscal year of the workpaper, the pair is flagged for human review before being included in the training set.

```json
{
  "OMB_Circular_A-133": {
    "superseded_by": "2 CFR 200",
    "effective_date": "2015-12-26",
    "replacement_id": "2 CFR 200"
  }
}
```

---

### Step 4 — Chunker (Long Context Resolution)

**Purpose:** Split large workpapers into token-bounded chunks that fit within the model's context window while preserving semantic boundaries.

**Context window budget:**

| Component | Tokens |
|---|---|
| System prompt + engagement metadata | ~500 |
| Active chunk (current workpaper section) | ~2,000 |
| RAG context (5 retrieved chunks × 400t each) | ~2,000 |
| Consistency context (3 past findings × 300t each) | ~900 |
| Computed audit facts (math tools block) | ~400 |
| Prompt template per task_type | ~300 |
| Response budget | ~2,000 |
| Safety margin | ~300 |
| **Total used** | **~8,400 / 32,768 tokens** |

This leaves substantial headroom. The active chunk target is 2,000 tokens.

**Chunker mode selection:**

The Chunker operates in two modes. Mode is determined by the `workpaper_type` label set in `engagement_labels.csv` — available on the `Record` at this stage.

| `workpaper_type` | Chunker mode | `content_json.chunk_mode` |
|---|---|---|
| `bank_reconciliation`, `trial_balance` | Numeric | `"numeric"` |
| `financial_statements`, `analytical_procedure` | Numeric | `"numeric"` |
| All other types | Semantic | `"semantic"` |

**Numeric Chunker** (runs instead of semantic chunking for the above types):

```
Input: Excel sheet rows (already extracted with data_only=True)
    │
    ├── Assert file_type is Excel — log WARNING if not (these workpaper types should always be Excel)
    │
    ├── Identify numeric columns
    │     A column is numeric if >80% of non-empty cells parse as float after currency normalisation
    │
    ├── Identify label column
    │     Leftmost non-numeric column → account descriptions / line item names
    │
    ├── For each row: extract structured dict
    │     {"label": str, "amount": float, "currency": str, "row_type": str}
    │     row_type detection:
    │       subtotal_rows (flagged by Extractor)            → "subtotal"
    │       label matches "balance per bank*"               → "opening"
    │       label matches "balance per book*"               → "closing"
    │       amount > 0                                      → "addition"
    │       amount < 0                                      → "deduction"
    │
    ├── Pack into content_json.rows[]
    │
    ├── Apply same 2,000-token row-group chunking limit (never cut mid-row)
    │     Set content_json.chunk_mode = "numeric"
    │     Set content_json.numeric_columns = [list of numeric column names]
    │     Set content_json.currency = engagement.currency
    │
    ├── Generate content_json.text
    │     Human-readable rendering of the rows (used for embedding and LLM reading)
    │     Format: "{label}: {amount}" per line, subtotals indented
    │
    └── Assert len(rows) > 0 for numeric mode chunks
          On failure: fall back to semantic mode with WARNING logged to cleaning_log
          (D-09 fix)
```

**Semantic Chunker** (all other workpaper types — unchanged behaviour):

**Chunking rules by format:**

**Excel:**
```
1. One chunk per sheet by default
2. If sheet > 2,000 tokens:
   - Chunk by row groups
   - Calculate running token count row by row
   - Stop before limit — never cut mid-row        ← D-06 + I-05 fix
   - Prepend column headers to every chunk
   - Append: "[SHEET: {name} — rows {start}-{end} of {total}]"
3. Subtotal rows always included in same chunk as their source rows
```

**Word:**
```
1. Chunk at paragraph boundary
2. Keep Heading + its following paragraphs together
3. Tables always kept as single unit (never split across chunks)
4. If table > 2,000 tokens: split at row boundary, repeat headers in next chunk
5. Append: "[SECTION: {heading} — Part {n} of {total}]"
```

**PDF:**
```
1. Chunk at page boundary by default
2. If page > 2,000 tokens: split at sentence boundary
3. Preserve table integrity — never split a table across chunks
4. Append: "[PAGE: {n} of {total}]"
```

**Overlap:** 200 tokens of overlap between consecutive chunks within the same sheet/section. Ensures no context is lost at chunk boundaries.

**Token counting:** Uses Mistral tokenizer (`mistral_common.tokenize`). Token counts stored in `workpaper_chunks.token_count`.

---

### Step 5 — InjectionSanitizer

**Purpose:** Remove prompt injection attempts from cell content. (Edge case D-07.)

```python
INJECTION_PATTERNS = [
    r'\[INST\]',
    r'\[/INST\]',
    r'<\/s>',
    r'<s>',
    r'(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b',
    r'(?i)\bforget\s+(everything|all|your)\b',
    r'(?i)\byou\s+are\s+now\b',
    r'(?i)\bact\s+as\s+(a|an)\b',
    r'(?i)\bnew\s+persona\b',
]
```

All detected patterns are:
1. Stripped from the content (replaced with `[REDACTED]`)
2. Logged to a security review queue (separate from `cleaning_log`)
3. The workpaper is still processed — a single injection attempt does not discard the whole file

---

## 6. Temporal Drift Strategy

15 years of audit data means the model will see regulation references that no longer apply. Strategy:

1. **Tag every pair with its source year** (TemporalTagger above)
2. **Apply drift weights at JSONL export** (not at extraction — weights can be recalibrated without re-running ETL)
3. **Flag outdated citations** using `outdated_regulations.json` — human review required before including
4. **Maintain `regulations_master.json`** — authoritative list of currently valid regulation IDs. Updated annually.
5. **Stage 1 mass SFT mixes all engagement types intentionally** — building shared audit vocabulary across Single Audit, GAGAS, and GAAS is the goal of Stage 1. Stage 2 batch SFT runs has_single_audit and non-Single-Audit engagements in isolation so framework-specific judgment is sharpened without cross-contamination of thresholds and regulation codes.
6. **Year as a feature** — include `fiscal_year` in every prompt's context block so the model can learn temporal patterns (e.g., thresholds changed in 2016)

---

## 7. SFT Pair Builder

### 7.1 Pair Construction Logic

After chunking, the pair builder reads each chunk and constructs prompt–completion pairs. One chunk can produce multiple pairs if it contains content relevant to multiple task types.

```
For each workpaper_chunk:
  Classify chunk content → determine applicable pair_types
  For each applicable pair_type:
    Select prompt template for pair_type + jurisdiction
    Fill template with chunk content
    Compute quality_score (heuristic: citation count, completeness, length)
    pair_hash = sha256(prompt + completion).hexdigest()   # idempotency key

    # Idempotency check — skip if this exact pair already exists
    # (handles re-submissions without producing duplicate training pairs)
    INSERT INTO sft_training_pairs (..., pair_hash, ...)
      VALUES (..., :pair_hash, ...)
      ON CONFLICT (workpaper_id, pair_type, pair_hash) DO NOTHING

    If row was inserted (not skipped by conflict):
      INSERT MongoDB training_pairs_content
      # MongoDB insert only runs when PG insert succeeds — keeps stores in sync
```

### 7.2 Prompt Templates (Mistral [INST]/[/INST] format)

**Template: procedure_conclusion**
```
<s>[INST] You are an audit assistant for a licensed CPA firm. 
Client: {client_type}. GAGAS: {is_gagas}. Single Audit: {has_single_audit}. 
Fiscal year: {fiscal_year}.

WORKPAPER SECTION: {section}

{chunk_content}

Based on the above workpaper content, state the audit conclusion for this 
procedure. Include: population tested, sample size, exceptions found (if any), 
and your overall conclusion. Reference applicable standards. [/INST]
{expected_conclusion}</s>
```

**Template: risk_response**
```
<s>[INST] You are an audit assistant for a licensed CPA firm.
Client: {client_type}. GAGAS: {is_gagas}. Single Audit: {has_single_audit}.

IDENTIFIED RISK FACTOR:
{risk_description}

ENTITY CONTEXT:
{chunk_content}

Classify this risk (CRITICAL/HIGH/MEDIUM/LOW) and describe the appropriate 
audit response. Cite the relevant standard. [/INST]
{expected_response}</s>
```

**Template: finding_recommendation**
```
<s>[INST] You are an audit assistant for a licensed CPA firm.
Client: {client_type}. GAGAS: {is_gagas}. Single Audit: {has_single_audit}.

WORKPAPER CONTENT:
{chunk_content}

Draft a finding using the following structure:
Criteria: [the standard or requirement]
Condition: [what was actually found]
Cause: [why the condition exists]
Effect: [the impact or potential impact]
Recommendation: [management action required]

Cite the applicable regulation. [/INST]
{expected_finding}</s>
```

**Template: analytical_narrative**
```
<s>[INST] You are an audit assistant for a licensed CPA firm.
Client: {client_type}. GAGAS: {is_gagas}. Single Audit: {has_single_audit}.

ANALYTICAL DATA:
{chunk_content}

Write a concise analytical narrative for partner review. Highlight key figures, 
significant variances (>10%), and any items requiring audit attention. [/INST]
{expected_narrative}</s>
```

### 7.3 Engagement-Level Split Assignment

```python
import hashlib

def assign_split(engagement_id: str) -> str:
    """
    Deterministic split assignment based on engagement_id hash.
    All workpapers from one engagement land in the same split.
    """
    h = int(hashlib.md5(engagement_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return 'train'
    elif h < 85:
        return 'val'
    else:
        return 'test'
```

**Why deterministic hash?** The same engagement will always get the same split, even if re-processed. No randomness that could cause inconsistency across pipeline runs.

### 7.4 Task Distribution Enforcement

Before JSONL export, verify no task exceeds 40% of total pairs:

```python
from collections import Counter

def verify_task_balance(pairs: list[dict]) -> None:
    counts = Counter(p['pair_type'] for p in pairs)
    total = len(pairs)
    for task, count in counts.items():
        pct = count / total
        if pct > 0.40:
            raise ValueError(
                f"Task imbalance: {task} is {pct:.1%} of pairs. "
                f"Max allowed: 40%. Resample before training."
            )
```

---

## 8. JSONL Export Format

Each line in the exported JSONL files is exactly this structure. No deviations.

```json
{
  "prompt": "<s>[INST] ...full instruction + workpaper content... [/INST]",
  "completion": "...expected model output...",
  "metadata": {
    "pair_id": "uuid",
    "pair_type": "procedure_conclusion",
    "client_type": "Govt",
    "is_gagas": true,
    "has_single_audit": true,
    "fiscal_year": 2022,
    "framework_section": "2 CFR 200.302",
    "split": "train",
    "drift_weight": 0.85
  }
}
```

**Critical formatting rules (T-02 fix):**
- Every `prompt` field starts with `<s>[INST]` and ends with `[/INST]`
- Every `completion` field ends with `</s>`
- A single `format_pair()` function generates 100% of training pairs — never manually formatted
- The formatter is tested on 10 examples with raw token ID inspection before any training run

---

## 9. All Edge Cases — Resolution Map

### Data Edge Cases

| ID | Severity | Problem | Resolution in this pipeline |
|---|---|---|---|
| D-01 | HIGH | Empty cells → NaN → "null" string in training | `df.fillna('')` in Excel extractor. Numeric cols: `df.fillna(0)`. |
| D-02 | HIGH | Merged cells → None in non-top-left cells | openpyxl `merged_cells` detection + forward-fill before pandas read |
| D-03 | CRITICAL | Formula cells return formula string not value | `openpyxl.load_workbook(data_only=True)` — always, no exceptions |
| D-04 | CRITICAL | pd.read_excel reads Sheet1 only | `pd.ExcelFile(path)` + iterate `xl.sheet_names` — all sheets always |
| D-05 | HIGH | Mixed dtypes in columns → silent truncation | `pd.read_excel(dtype=str)` always — normalize in Normalizer step |
| D-06 | CRITICAL | Full workpaper overflows context window | Chunker truncates at row/sentence boundary, appends `[TRUNCATED]` marker |
| D-07 | CRITICAL | Prompt injection in cell values | InjectionSanitizer strips all known injection patterns, logs to security queue |
| D-08 | HIGH | Same workpaper in both train and test splits | Deterministic engagement-level hash split assignment — never row-level |
| D-09 | HIGH | Numeric workpaper chunk has empty rows[] — math tools receive no input | Numeric Chunker asserts `len(rows) > 0`. On failure: log WARNING to cleaning_log, fall back to semantic mode so chunk is not silently discarded |

### Training Edge Cases

| ID | Severity | Problem | Resolution |
|---|---|---|---|
| T-01 | CRITICAL | No pad token → misaligned attention masks | `tokenizer.pad_token = tokenizer.eos_token` + `padding_side='right'` — asserted in pre-flight |
| T-02 | CRITICAL | Wrong instruction template format | Single `format_pair()` function used for 100% of pairs. Token ID inspection before training. |
| T-03 | HIGH | Catastrophic forgetting of general reasoning | 5-10% general instruction data mixed into training set. LoRA rank=16 (not 64). LR ≤ 2e-4. |
| T-04 | HIGH | Imbalanced task distribution | `verify_task_balance()` — max 40% per task enforced at export, JSONL export blocked if exceeded |
| T-05 | HIGH | LR too high → loss spikes / NaN | LR = 2e-4 fixed. Cosine scheduler + warmup_ratio=0.03. Reduce 50% and restart from last clean checkpoint on spike. |
| T-06 | HIGH | Overfitting (val loss rises, train loss falls) | `load_best_model_at_end=True`, patience=3 eval steps, checkpoints every 200 steps |
| T-07 | CRITICAL | VRAM OOM mid-training due to fragmentation | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` + `gradient_checkpointing=True` |
| T-08 | MEDIUM | Inconsistent response lengths across tasks | Length normalization per task type at pair construction. Outlier filtering before dataset. |

### Inference Edge Cases

| ID | Severity | Problem | Resolution |
|---|---|---|---|
| I-01 | CRITICAL | Hallucinated regulation numbers | Post-processor verifies every citation against `regulations_master.json`. Unverified → flagged, not surfaced. |
| I-02 | CRITICAL | LLM arithmetic errors on thresholds and numeric workpaper data | All arithmetic done by `model/tools/` library (thresholds, materiality, bank_rec, trial_balance, ratios, variance). LangGraph dispatches tools before generation. Results injected as `COMPUTED AUDIT FACTS` block. Model explains; it does not calculate. |
| I-03 | HIGH | Non-deterministic risk ratings | `temperature=0.1` for classification/structured tasks. `do_sample=False` for risk_classification and compliance_check. |
| I-04 | HIGH | Ollama serial queue under concurrent load | vLLM AsyncLLMEngine for production. `max_num_seqs=16`. Ollama for local dev only. |
| I-05 | HIGH | Context overflow truncation mid-row | Chunker ensures truncation only at complete row/sentence boundary. `[DOCUMENT TRUNCATED]` marker appended. |
| I-06 | HIGH | Missing required fields in output | Post-processor validates all required fields. One retry with explicit field reminder. Error surfaced to auditor if still missing. |
| I-07 | HIGH | flash-attn CUDA version mismatch | `pip install flash-attn --no-build-isolation` compiled for CUDA 12.2. Startup version check assertion. |

### DPO Edge Cases

| ID | Severity | Problem | Resolution |
|---|---|---|---|
| P-01 | CRITICAL | Junior corrections used as DPO ground truth | Tier gate: only `partner` or `senior` corrections eligible. Others → review queue. |
| P-02 | HIGH | Identical chosen/rejected pairs | `difflib.SequenceMatcher` similarity check. Pairs with similarity > 0.95 removed before every DPO run. |
| P-03 | HIGH | Reference model drift across DPO cycles | `beta=0.1–0.3`, max 2 DPO epochs per cycle. General capability eval after every DPO cycle. |
| P-04 | MEDIUM | Feedback only on easy cases | Confidence-score-based routing: low-confidence outputs surfaced to senior reviewers first. |
| P-05 | HIGH | Too few DPO pairs → overfitting | Hard minimum of 300 pairs. DAG gate blocks run if count < 300. |

### Infrastructure Edge Cases

| ID | Severity | Problem | Resolution |
|---|---|---|---|
| N-01 | CRITICAL | Checkpoint corruption on crash | Validate every checkpoint after save: load + run 5 fixed test examples. Never resume from unvalidated checkpoint. |
| N-02 | CRITICAL | Adapter loading order mismatch | Fixed loading function: base model → CPU → quantize → `.to(device)` → `prepare_model_for_kbit_training()` → load adapter. Step assertions enforced. |
| N-03 | HIGH | bfloat16 / float16 mismatch | Standardise: `bnb_4bit_compute_dtype=torch.bfloat16` + `bf16=True` + `fp16=False` everywhere. Never mix. |
| N-04 | HIGH | Disk full → MLflow silently stops saving | Pre-flight: verify disk space > 3× expected checkpoint size. 80% utilisation alert. MLflow configured to raise on failed artifact saves. |
| N-05 | CRITICAL | VLAN misconfiguration → external call | Egress test before any real data: `curl` from inference server to external IP → must be blocked. Run after every infra change. |

### Evaluation Edge Cases

| ID | Severity | Problem | Resolution |
|---|---|---|---|
| E-01 | HIGH | Holdout set has only seen workpaper types | Test set must include at least one example from every workpaper_type ENUM value. Stress test set of unusual workpapers maintained separately. |
| E-02 | HIGH | ROUGE misleads on structured outputs | Field-level accuracy for structured fields (Risk, Finding, Regulation, Recommendation). ROUGE only for free-text summary fields. |
| E-03 | MEDIUM | Reviewer fatigue bias | Max 50 outputs per auditor per session. 5 auditors spread across sessions. Gold standard negative outputs inserted at random positions. |
| E-04 | MEDIUM | Perplexity drops but task accuracy flat | Perplexity never used as primary metric. Primary: F1 per task + hallucination_rate + format_compliance + auditor approval rate. |

---

## 10. Pre-Training Checklist

Run programmatically in `training_trigger` DAG before launching any training run. Maps to edge case IDs above.

| # | Check | Maps To |
|---|---|---|
| 1 | `engagement_labels.csv` present and parsed for every engagement in training set; unlabelled engagements excluded | — |
| 2 | `tokenizer.pad_token = tokenizer.eos_token` is set | T-01 |
| 2 | `tokenizer.padding_side = 'right'` is set | T-01 |
| 3 | All training pairs formatted via `format_pair()` — no manual formatting | T-02 |
| 4 | Excel extraction used `data_only=True` | D-03 |
| 5 | All Excel sheets processed (not just Sheet1) | D-04 |
| 6 | Empty cells filled with `''` not NaN | D-01 |
| 7 | Merged cells forward-filled before JSON | D-02 |
| 8 | `dtype=str` on all `pd.read_excel()` calls | D-05 |
| 9 | InjectionSanitizer applied to all cell content | D-07 |
| 10 | Split assigned by engagement hash, not row | D-08 |
| 10a | Numeric workpaper types (bank_reconciliation, trial_balance, financial_statements, analytical_procedure) chunked in numeric mode; content_json.rows[] non-empty | D-09 |
| 11 | Task distribution verified — no task > 40% | T-04 |
| 12 | Response lengths normalized per task type | T-08 |
| 13 | 5-10% general instruction data in training set | T-03 |
| 14 | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` set | T-07 |
| 15 | `gradient_checkpointing=True` in TrainingArguments | T-07 |
| 16 | `bf16=True` and `fp16=False` in TrainingArguments | N-03 |
| 17 | `bnb_4bit_compute_dtype=torch.bfloat16` in quant config | N-03 |
| 18 | Disk space > 3× expected checkpoint size | N-04 |
| 19 | VLAN egress block verified with curl test | N-05 |
| 20 | MLflow tracking server running and accessible | N-04 |

---

*Next document: [05_rag_inference.md](05_rag_inference.md) — RAG & Inference Design*
