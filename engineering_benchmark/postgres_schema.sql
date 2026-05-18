-- =============================================================================
-- AuditAI PostgreSQL Schema
-- engineering_benchmark/postgres_schema.sql
-- =============================================================================
-- Four tables per architecture doc (Slide 4):
--   engagements   — client engagement records
--   workpapers    — individual workpaper metadata linked to engagements
--   sop_chunks    — chunked SOP text with versioning for re-embed support
--   findings      — audit findings linked to workpapers
--
-- Run with:
--   docker exec -i postgres psql -U auditai -d auditai < engineering_benchmark/postgres_schema.sql
--
-- Idempotent — safe to run multiple times (CREATE TABLE IF NOT EXISTS).
-- =============================================================================


-- ---------------------------------------------------------------------------
-- engagements
-- ---------------------------------------------------------------------------
-- One row per client engagement (audit year + client combination).
-- All workpapers and findings link back here.

CREATE TABLE IF NOT EXISTS engagements (
    id                  SERIAL PRIMARY KEY,
    engagement_code     VARCHAR(64)  NOT NULL UNIQUE,
    client_name         VARCHAR(255) NOT NULL,
    client_type         VARCHAR(64)  NOT NULL,   -- NPO, government, for_profit, tribal
    fiscal_year_end     DATE,
    is_gagas            BOOLEAN      NOT NULL DEFAULT FALSE,
    has_single_audit    BOOLEAN      NOT NULL DEFAULT FALSE,
    engagement_partner  VARCHAR(128),
    preparer_id         VARCHAR(64),
    reviewer_id         VARCHAR(64),
    engagement_decision VARCHAR(32),             -- accepted, not_accepted, continued
    status              VARCHAR(32)  NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_engagements_client_name
    ON engagements (client_name);

CREATE INDEX IF NOT EXISTS idx_engagements_fiscal_year_end
    ON engagements (fiscal_year_end);

CREATE INDEX IF NOT EXISTS idx_engagements_client_type
    ON engagements (client_type);


-- ---------------------------------------------------------------------------
-- workpapers
-- ---------------------------------------------------------------------------
-- One row per workpaper file processed through the ETL pipeline.
-- Links to engagements. Stores extraction metadata and training gate flags.

CREATE TABLE IF NOT EXISTS workpapers (
    id                      SERIAL PRIMARY KEY,
    engagement_id           INTEGER      REFERENCES engagements(id) ON DELETE SET NULL,
    file_name               VARCHAR(512) NOT NULL,
    file_type               VARCHAR(32)  NOT NULL,  -- docx, pdf_text, pdf_scanned, xlsx, csv, json
    file_hash               CHAR(64)     NOT NULL UNIQUE,  -- SHA-256 of raw file bytes
    file_size_bytes         BIGINT,
    source_path             TEXT,
    extraction_method       VARCHAR(64),
    extraction_status       VARCHAR(32)  NOT NULL DEFAULT 'pending',  -- success, partial, failed, pending
    extraction_confidence   FLOAT        NOT NULL DEFAULT 0.0,
    ocr_used                BOOLEAN      NOT NULL DEFAULT FALSE,
    word_count              INTEGER,
    page_count              INTEGER,
    pii_scrubbed            BOOLEAN      NOT NULL DEFAULT FALSE,
    auditor_approved        BOOLEAN      NOT NULL DEFAULT FALSE,
    reviewer_id             VARCHAR(64),
    review_date             DATE,
    needs_review            BOOLEAN      NOT NULL DEFAULT FALSE,
    fields_present          TEXT[],                -- array of canonical field names found
    fields_missing          TEXT[],                -- array of canonical field names not found
    workpaper_type          VARCHAR(128),          -- e.g. bank_reconciliation, trial_balance
    extracted_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workpapers_engagement_id
    ON workpapers (engagement_id);

CREATE INDEX IF NOT EXISTS idx_workpapers_file_hash
    ON workpapers (file_hash);

CREATE INDEX IF NOT EXISTS idx_workpapers_extraction_status
    ON workpapers (extraction_status);

CREATE INDEX IF NOT EXISTS idx_workpapers_auditor_approved
    ON workpapers (auditor_approved);

CREATE INDEX IF NOT EXISTS idx_workpapers_workpaper_type
    ON workpapers (workpaper_type);


-- ---------------------------------------------------------------------------
-- sop_chunks
-- ---------------------------------------------------------------------------
-- One row per SOP text chunk stored in Qdrant.
-- chunks_version allows re-embedding when SOP documents are updated.
-- The qdrant_point_id links this row to the Qdrant vector.

CREATE TABLE IF NOT EXISTS sop_chunks (
    id                  SERIAL PRIMARY KEY,
    qdrant_point_id     VARCHAR(64)  NOT NULL UNIQUE,  -- sha256(source_doc + char_start)
    source_doc          VARCHAR(512) NOT NULL,          -- original SOP filename
    sop_version         VARCHAR(32)  NOT NULL,          -- e.g. "2024-Q1", "v3.2"
    chunks_version      INTEGER      NOT NULL DEFAULT 1, -- increment on re-embed
    chunks_hash         CHAR(64),                       -- sha256 of chunk text
    section_prefix      VARCHAR(256),                   -- "SOP §3.1 — Reconciliation: "
    content             TEXT         NOT NULL,           -- full chunk text
    char_start          INTEGER      NOT NULL,           -- char offset in source doc
    char_end            INTEGER      NOT NULL,
    token_count         INTEGER,
    workpaper_type      VARCHAR(128),                   -- which workpaper type this applies to
    year                INTEGER,                        -- fiscal year if applicable
    is_rollforward      BOOLEAN      NOT NULL DEFAULT FALSE,
    embedded_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_source_doc
    ON sop_chunks (source_doc);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_sop_version
    ON sop_chunks (sop_version);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_chunks_version
    ON sop_chunks (chunks_version);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_workpaper_type
    ON sop_chunks (workpaper_type);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_qdrant_point_id
    ON sop_chunks (qdrant_point_id);


-- ---------------------------------------------------------------------------
-- findings
-- ---------------------------------------------------------------------------
-- One row per AI-drafted or auditor-written finding.
-- Links to workpapers. Tracks the full lifecycle from AI draft to final.

CREATE TABLE IF NOT EXISTS findings (
    id                  SERIAL PRIMARY KEY,
    workpaper_id        INTEGER      REFERENCES workpapers(id) ON DELETE CASCADE,
    engagement_id       INTEGER      REFERENCES engagements(id) ON DELETE SET NULL,
    finding_text        TEXT         NOT NULL,
    recommendation_text TEXT,
    sop_sections        TEXT[],                  -- e.g. ARRAY['§3.1', '§4.2']
    severity            VARCHAR(32),             -- material, significant, minor, informational
    status              VARCHAR(32)  NOT NULL DEFAULT 'ai_draft',
                                                 -- ai_draft, under_review, finalized, dismissed
    pair_type           VARCHAR(32),             -- clean, deficient
    stage               VARCHAR(32),             -- stage2, stage3
    is_gagas_finding    BOOLEAN      NOT NULL DEFAULT FALSE,
    client_type         VARCHAR(64),
    auditor_id          VARCHAR(64),             -- who finalized this finding
    ai_model            VARCHAR(128),            -- which model drafted this
    extraction_confidence FLOAT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_findings_workpaper_id
    ON findings (workpaper_id);

CREATE INDEX IF NOT EXISTS idx_findings_engagement_id
    ON findings (engagement_id);

CREATE INDEX IF NOT EXISTS idx_findings_status
    ON findings (status);

CREATE INDEX IF NOT EXISTS idx_findings_severity
    ON findings (severity);

CREATE INDEX IF NOT EXISTS idx_findings_stage
    ON findings (stage);


-- ---------------------------------------------------------------------------
-- updated_at trigger function
-- ---------------------------------------------------------------------------
-- Automatically updates updated_at on every row update.

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_engagements_updated_at
    BEFORE UPDATE ON engagements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_workpapers_updated_at
    BEFORE UPDATE ON workpapers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_sop_chunks_updated_at
    BEFORE UPDATE ON sop_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_findings_updated_at
    BEFORE UPDATE ON findings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();