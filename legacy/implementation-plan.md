# Horse ID Cloud Platform — Implementation Plan

## Context

Migrating the Horse ID system from a local CLI pipeline (CSV files, local scripts, S3+pickle features) to a serverless cloud platform. The new system uses Google Drive as the photo source of truth, Neon Postgres+pgvector as the database, AWS Lambda for all compute, and a React SPA for the web interface. The existing SMS/Twilio identification flow continues but reads from the new database instead of S3 CSVs.

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 - Migration to Drive | **Complete** | |
| 2 - Foundation | **Complete** | 3 migrations. Changes API + incremental sync added. |
| 3 - ML Workers | **Complete** | All workers + `syncer.py` and `lambda_utils.py`. |
| 4 - API Routes | **Complete** | 7 routes. Backend runs as Express server (not yet Lambda). |
| 5 - SMS Flow | **Complete** | Existing responder stayed in project root. No `cloud/sms/` needed. |
| 6 - WebSocket | **Not yet implemented** | Using 2-second HTTP polling for now. |
| 7 - Frontend | **Complete** | Dashboard, HerdDetail, HorseDetail, Identify pages. |
| 8 - Deployment | **Partial** | No SAM template. CodeBuild + ECR + manual Lambda config. |
| 9 - Production Cutover | **In Progress** | Pipeline running. Twilio webhook pointed at responder Lambda. |
| 10 - Legacy Cleanup | **Not started** | All legacy scripts still in project root. |

### Key deviations from original plan

1. **No SAM template** — Infrastructure managed via AWS console + CodeBuild, not IaC
2. **No `cloud/sms/` directory** — Responder stayed in project root (`Dockerfile.responder`, `webhook_responder.py`)
3. **No WebSocket** — HTTP polling works well enough
4. **No Better Auth** — Not yet implemented
5. **Backend is Express server, not Lambda** — Runs locally / on EC2, not behind API Gateway yet
6. **2 Lambda functions, not 4** — `twilio-webhook-responder` + `horse-id-ml-worker` only
7. **Event-driven pipeline** — Added post-plan. Uses Drive Changes API for incremental sync, Lambda self-chaining for detect→extract. Replaces batch-oriented approach.

### Known gaps

- **Full sync doesn't chain to detection** — `runFullSync()` inserts photos as `pending` but doesn't fan out to Lambda. Incremental sync works end-to-end.

## Tech Decisions

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Frontend | **React + Vite + TypeScript** | Simple SPA, no SSR needed. Vite is fast, minimal config |
| API framework | **Node.js + Express** | Same ecosystem as React. Runs as standalone server for now |
| ML workers | **Python + Lambda** | Existing ML code is Python (torch, timm, ultralytics, wildlife-tools). Separate container image |
| IaC | ~~AWS SAM~~ **CodeBuild + manual** | SAM deferred. CodeBuild builds/pushes images, updates Lambdas |
| Google Drive | **googleapis** (Node, for API/sync) + **google-api-python-client** (Python, for ML workers downloading images) | Each runtime uses its native SDK |
| Real-time updates | ~~API Gateway WebSocket~~ **HTTP polling** | Not yet implemented. Using 2-second polling for now |
| Auth | ~~Better Auth~~ **None (deferred)** | Not yet implemented |
| CSS | **Tailwind** | Fast to build, responsive out of the box |
| ORM | **Drizzle** | Lightweight, TypeScript-native, works well with Neon's serverless driver |

## Project Structure

```
cloud/
├── docker-compose.yml           # Local dev (Postgres+pgvector)
├── .env
│
├── frontend/                    # React + Vite + TypeScript
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts        # API client (typed fetch wrappers)
│   │   ├── components/Layout.tsx
│   │   └── pages/
│   │       ├── Dashboard.tsx     # Stats, sync/process controls, error list
│   │       ├── HerdDetail.tsx    # Horses in a herd
│   │       ├── HorseDetail.tsx   # Photos for a horse, exclude toggle
│   │       └── Identify.tsx      # Upload image, see matches
│   └── vite.config.ts
│
├── backend/                     # Node.js + Express (standalone server)
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── app.ts               # Express app setup
│   │   ├── routes/
│   │   │   ├── sync.ts          # POST /sync, GET /sync/:id
│   │   │   ├── process.ts       # POST /process (resetStuck + fanOut)
│   │   │   ├── stats.ts         # GET /stats (photo counts + sync status)
│   │   │   ├── herds.ts         # GET /herds
│   │   │   ├── horses.ts        # GET /herds/:id/horses, GET /horses/:id
│   │   │   ├── photos.ts        # PATCH /photos/:id, GET /photos/errors, GET /photos/:id/image
│   │   │   └── identify.ts      # POST /identify (upload → Lambda → results)
│   │   ├── db/
│   │   │   ├── schema.ts        # Drizzle schema definitions
│   │   │   ├── client.ts        # Neon serverless driver connection
│   │   │   └── migrate.ts       # Migration runner
│   │   └── services/
│   │       ├── drive.ts         # Google Drive API (list, changes, download proxy)
│   │       ├── sync.ts          # Sync orchestration (Changes API + full scan fallback)
│   │       ├── process.ts       # fanOutProcessing + resetStuckPhotos
│   │       └── config.ts        # Config loader (.env)
│
├── workers/                     # Python ML workers (Lambda container)
│   ├── requirements.txt
│   ├── Dockerfile               # ML image (torch, YOLO, wildlife-mega-L-384)
│   ├── handler.py               # Lambda entry: routes detect/extract/sync_batch/identify
│   ├── detector.py              # YOLO classification → chains to extraction
│   ├── extractor.py             # Feature extraction → pgvector
│   ├── identifier.py            # Query image → similarity search
│   ├── syncer.py                # Batch sync: upsert herds/horses/photos → chain to detect
│   ├── lambda_utils.py          # Cross-Lambda invocation helper
│   ├── drive_client.py          # Download images from Drive
│   └── db.py                    # psycopg2 + pgvector
│
├── db/
│   └── migrations/
│       ├── 001_initial.sql      # Schema + pgvector extension
│       ├── 002_sync_progress.sql # Sync progress columns
│       └── 003_drive_sync_state.sql # Changes API token + photos.updated_at
│
└── scripts/
    └── migrate_to_drive.py      # One-time: populate Drive from existing data

# Project root (not in cloud/):
├── Dockerfile.responder         # Twilio webhook responder image
├── webhook_responder.py         # SMS handler (invokes ml-worker for identify)
└── buildspec.yml                # CodeBuild: build images → ECR → update Lambdas
```

## Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE herds (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    drive_folder_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    herd_id INTEGER NOT NULL REFERENCES herds(id) ON DELETE CASCADE,
    drive_folder_id TEXT NOT NULL UNIQUE,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (herd_id, name)
);

CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    horse_id INTEGER NOT NULL REFERENCES horses(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    drive_file_id TEXT NOT NULL UNIQUE,
    drive_md5 TEXT,
    processing_status TEXT DEFAULT 'pending',  -- pending, detecting, detected, extracting, ready, error
    detection_result TEXT,                      -- NULL, NONE, SINGLE, MULTIPLE
    excluded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()        -- for stuck photo detection
);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at() RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;
CREATE TRIGGER photos_updated_at BEFORE UPDATE ON photos
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    photo_id INTEGER NOT NULL UNIQUE REFERENCES photos(id) ON DELETE CASCADE,
    horse_id INTEGER NOT NULL REFERENCES horses(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    extracted_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON features USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE TABLE sync_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    herds_total INTEGER DEFAULT 0,
    herds_scanned INTEGER DEFAULT 0,
    last_heartbeat TIMESTAMPTZ,
    files_scanned INTEGER DEFAULT 0,
    files_added INTEGER DEFAULT 0,
    files_removed INTEGER DEFAULT 0,
    files_moved INTEGER DEFAULT 0
);

CREATE TABLE drive_sync_state (
    id INT PRIMARY KEY DEFAULT 1,
    changes_token TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

## Lambda Functions (2 container images)

| Function | Runtime | Image | Memory | Timeout | Trigger |
|----------|---------|-------|--------|---------|---------|
| **horse-id-ml-worker** | Python | `cloud/workers/Dockerfile` | 4096 MB | 15 min | Async invoke from backend or self-chain |
| **twilio-webhook-responder** | Python | `Dockerfile.responder` | 256 MB | 10s | API Gateway HTTP (Twilio webhook) |

The **ml-worker** handles detection, extraction, identification, AND sync batches — the handler inspects `event.task` to route to the right function. Tasks: `detect`, `extract`, `identify`, `sync_batch`. Self-chains detect→extract for SINGLE photos.

The **webhook-responder** is the original responder with minimal changes. Invokes ml-worker for identification.

The **backend** runs as a standalone Express server (not a Lambda). It handles Drive API calls, sync orchestration, and serves the frontend API.

## Implementation Phases

### Phase 1: Migration to Drive — COMPLETE
**Populate Google Drive with existing data so all subsequent phases have real data to work with.**

1. Set up Google Drive service account + share root folder
2. `cloud/scripts/migrate_to_drive.py`:
   - Read existing `horse_herds.csv` + merged manifest CSV
   - Create top-level folder per herd in Drive
   - Create horse subfolder per horse within each herd
   - Upload photos from S3/local storage into the appropriate folders
   - Log mapping of Drive folder/file IDs for verification
3. Verify: folder structure matches expected herds/horses/photo counts

### Phase 2: Foundation — COMPLETE
**Database + Drive sync + local dev environment. Tested against real Drive data from Phase 1.**

1. Neon project with pgvector. 3 migrations (001 initial, 002 sync progress, 003 drive sync state + updated_at)
2. `cloud/backend/src/db/` — Drizzle schema + Neon serverless driver
3. `cloud/backend/src/services/drive.ts` — Google Drive API client:
   - `listFolders()`, `listImageFiles()` — for full scan
   - `getStartPageToken()`, `getChanges()` — for Changes API incremental sync
4. `cloud/backend/src/services/sync.ts` — sync orchestration:
   - `runIncrementalSync()` — Changes API, dispatches `sync_batch` to Lambda
   - `runFullSync()` — full Drive tree walk, direct DB upserts (first-time fallback)
   - Track progress in sync_runs table
5. `cloud/docker-compose.yml` — Postgres+pgvector for local dev

### Phase 3: ML Workers — COMPLETE
**Detection + extraction running in Lambda, writing to pgvector.**

1. `cloud/workers/Dockerfile` — Python 3.13 Lambda base, YOLO + wildlife-mega-L-384 pre-downloaded
2. `cloud/workers/handler.py` — routes `event.task` to `detect`, `extract`, `identify`, `sync_batch`
3. `cloud/workers/detector.py` — YOLO detection, marks NONE/SINGLE/MULTIPLE, self-chains to extract for SINGLE
4. `cloud/workers/extractor.py` — wildlife-mega-L-384 embeddings → pgvector, marks photo `ready`
5. `cloud/workers/identifier.py` — extract query embedding → pgvector cosine similarity → top-N results
6. `cloud/workers/syncer.py` — *(added post-plan)* batch sync: upsert herds/horses/photos, chain to detect
7. `cloud/workers/lambda_utils.py` — *(added post-plan)* cross-Lambda invocation helper
8. `cloud/workers/db.py` — psycopg2 + pgvector, `update_photo_status()` sets `updated_at`
9. `cloud/workers/drive_client.py` — download images from Drive by file ID

### Phase 4: API Routes — COMPLETE
**Express API serving data and triggering operations.**

> **Deviation:** Backend runs as standalone Express server, not Lambda behind API Gateway. No `handler.ts` Lambda wrapper. No auth routes (Better Auth deferred).

1. `cloud/backend/src/app.ts` — Express setup + middleware + route registration
2. Routes:
   - `POST /api/sync` → triggers incremental sync, returns sync_run ID
   - `GET /api/sync/:id` → sync run status
   - `POST /api/process` → resetStuckPhotos + fanOutProcessing (detect + extract)
   - `GET /api/stats` → photo counts by status + sync status
   - `GET /api/herds` → list herds with horse/photo counts
   - `GET /api/herds/:id/horses` → horses in herd
   - `GET /api/horses/:id` → horse detail with photos
   - `PATCH /api/photos/:id` → toggle exclude
   - `GET /api/photos/errors` → list error photos
   - `GET /api/photos/:id/image` → proxy image from Drive
   - `POST /api/identify` → upload image → invoke ml-worker → return matches
3. `cloud/backend/src/services/process.ts` — `fanOutProcessing()` + `resetStuckPhotos()`

### Phase 5: SMS Flow — COMPLETE (no changes needed)
**Existing Twilio SMS backed by Neon.**

> **Deviation:** The `cloud/sms/` directory was never created. The existing `webhook_responder.py` and `Dockerfile.responder` in the project root already worked — they just needed the ml-worker Lambda name updated. Built and deployed via CodeBuild alongside the ml-worker.

1. `webhook_responder.py` (project root) — receives Twilio webhook, invokes ml-worker with `task: 'identify'`
2. `Dockerfile.responder` (project root) — lightweight image: boto3 + twilio
3. ML worker's `identifier.py` handles identification (shared with web flow)

### Phase 6: WebSocket Progress — NOT YET IMPLEMENTED
**Real-time updates during sync and processing.**

> Not yet implemented. Currently using 2-second HTTP polling (`setInterval` in Dashboard.tsx).

### Phase 7: Frontend — COMPLETE
**React SPA.**

> **Deviation:** No auth pages (Better Auth deferred). No WebSocket (polling instead). Components are inlined in pages rather than separate files.

1. **Pages:**
   - `/` — Dashboard: sync/process buttons, stats bar, error list, herd cards
   - `/herds/:id` — horses in herd with photo counts + thumbnails
   - `/horses/:id` — photo grid with exclude toggle
   - `/identify` — upload image, optional herd filter, ranked results
2. **Components:** `Layout.tsx` (nav wrapper)
3. Photo display: all images proxied via `GET /api/photos/:id/image`
4. Polling: 2-second interval while sync running or photos processing

### Phase 8: Deployment — PARTIAL
**CI/CD via CodeBuild.**

> **Deviation:** No SAM template. Infrastructure created manually via AWS console. CodeBuild handles image builds and Lambda updates only.

1. `buildspec.yml` — CodeBuild pipeline:
   - Builds `Dockerfile.responder` → `responder-latest` tag
   - Builds `cloud/workers/Dockerfile` → `ml-worker-latest` tag
   - Pushes both to ECR (`horse-id-lambda-repo`)
   - Updates Lambda function code for `twilio-webhook-responder` and `horse-id-ml-worker`
2. Lambda functions, IAM roles, API Gateway configured manually in AWS console
3. Backend runs as standalone Express server (not deployed via CI/CD yet)

### Phase 9: Production Cutover — IN PROGRESS
**Switch live traffic to new system.**

1. ~~Deploy full stack via SAM~~ Lambda functions deployed manually + CodeBuild
2. ✅ Production sync run — ~5000+ photos synced from Drive
3. ✅ Detection + extraction pipeline running (event-driven)
4. ✅ Twilio webhook pointed at `twilio-webhook-responder` Lambda
5. Remaining: verify identification accuracy, deprecate old S3 data

### Phase 10: Legacy Cleanup — NOT STARTED
**Move old CLI/CSV pipeline code out of the way.**

All legacy scripts still in project root. Move to `legacy/` when ready:
- `ingest_from_dir.py`, `ingest_from_email.py` — old ingestion scripts
- `normalize_horse_names.py` — CLI name normalization (replaced by Drive folder names)
- `multi_horse_detector.py` — local detection script (replaced by ml-worker Lambda)
- `merge_horse_identities.py` — identity merging (no longer needed)
- `extract_features.py` — local feature extraction (replaced by ml-worker Lambda)
- `upload_to_s3.py` — S3 CSV/pickle upload (replaced by Neon)
- `generate_gallery.py` — static HTML galleries (replaced by web UI)
- `manage_horses.py`, `review_merges_app.py` — Streamlit apps (replaced by web UI)
- `run_pipeline.py`, `pipeline_lock.py` — pipeline orchestration (replaced by sync)
- `parse_horse_herds.py` — Excel parser (herds now from Drive folders)
- `test_lambda_app.py` — old Lambda test UI
- `Dockerfile.horse_id` — old ML image (replaced by `cloud/workers/Dockerfile`)
- `horse-id-requirements.txt`, `responder-requirements.txt` — old requirements

Keep in project root (still used):
- `horse_detection_lib.py` — detection logic reused by ml-worker
- `config.yml` — detection thresholds referenced by workers
- `webhook_responder.py`, `Dockerfile.responder` — still actively deployed
- `CLAUDE.md` — update to reflect new architecture

## Event-Driven Pipeline (added post-plan)

Added after the original plan was written. Replaces the batch-oriented sync→detect→extract approach.

### Architecture

```
User clicks Sync
  → Backend calls Drive Changes API (or full scan if first time)
  → Groups changed files into batches
  → Invokes ml-worker Lambda with task: sync_batch (async, per batch)

sync_batch Lambda (per batch of changed files)
  → Upserts herds/horses/photos in DB
  → Invokes ml-worker Lambda with task: detect (for new/modified photos)

detect Lambda (per batch)
  → Runs YOLO on each photo
  → For SINGLE photos, self-invokes with task: extract

extract Lambda (per batch)
  → Extracts features, inserts into pgvector
  → Photo is "ready"
```

Each stage chains to the next. No orchestrator, no timers, no polling loops.

### Recovery

`POST /api/process` (Reprocess button):
1. `resetStuckPhotos()` — resets photos stuck in `detecting`/`extracting` >15 minutes
2. `fanOutProcessing("detect")` — dispatches pending photos to Lambda
3. `fanOutProcessing("extract")` — dispatches detected SINGLE photos to Lambda

### Known gap

`runFullSync()` (first-time sync with no Changes API token) inserts photos as `pending` but does NOT fan out to Lambda. Need to add `fanOutProcessing("detect")` at the end of full sync. Incremental sync works end-to-end.

## Testing Strategy

> **Status:** No automated tests written yet. Testing has been manual via the web UI and CloudWatch logs.

Planned integration tests (not yet implemented):

### `cloud/tests/workers/test_detector.py` (Pytest)
- Detection classifies known SINGLE/MULTIPLE/NONE images correctly
- Detection writes results to DB and updates processing_status

### `cloud/tests/workers/test_identifier.py` (Pytest)
- Feature extraction produces 384-dim vector, written to pgvector
- Identification query returns correct top-N matches
- Herd filter narrows results correctly

## Verification

**Current (manual):**
- Backend running locally, Lambdas in AWS
- Trigger sync from Dashboard → verify photos appear in DB
- Monitor CloudWatch for detect/extract Lambda invocations
- Browse herds/horses in web UI, verify photo counts
- Identify a known horse via Identify page
- Send test SMS, verify response

**Production:**
- CodeBuild pushes images to ECR and updates Lambdas
- Run migration via `cloud/backend/src/db/migrate.ts`
- Trigger sync, verify pipeline runs end-to-end
- Compare identification accuracy with previous system
