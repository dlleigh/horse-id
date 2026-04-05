# Horse ID Cloud Platform — Implementation Plan

## Context

Migrating the Horse ID system from a local CLI pipeline (CSV files, local scripts, S3+pickle features) to a serverless cloud platform. The new system uses Google Drive as the photo source of truth, Neon Postgres+pgvector as the database, AWS Lambda for all compute, and a React SPA for the web interface. The existing SMS/Twilio identification flow continues but reads from the new database instead of S3 CSVs.

## Tech Decisions

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Frontend | **React + Vite + TypeScript** | Simple SPA, no SSR needed. Vite is fast, minimal config |
| API framework | **Node.js + Express** | Same ecosystem as React + Better Auth. Runs in Lambda via `serverless-http` or direct handler |
| ML workers | **Python + Lambda** | Existing ML code is Python (torch, timm, ultralytics, wildlife-tools). Separate container image |
| IaC | **AWS SAM** | Lambda-centric, simpler than CDK for this scale |
| Google Drive | **googleapis** (Node, for API/sync) + **google-api-python-client** (Python, for ML workers downloading images) | Each runtime uses its native SDK |
| Real-time updates | **API Gateway WebSocket API** | Native AWS, Lambda-compatible |
| Auth | **Better Auth (Neon Auth)** | Built into Neon, native JS SDK for both Express and React |
| CSS | **Tailwind** | Fast to build, responsive out of the box |
| ORM | **Drizzle** | Lightweight, TypeScript-native, works well with Neon's serverless driver |

## Project Structure (new `cloud/` directory)

```
cloud/
├── template.yaml                # SAM template
├── docker-compose.yml           # Local dev
├── .env.example
│
├── frontend/                    # React + Vite + TypeScript
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/                 # API client (fetch wrappers)
│   │   ├── components/          # UI components
│   │   ├── pages/               # Route pages
│   │   └── hooks/               # useWebSocket, useAuth
│   └── vite.config.ts
│
├── backend/                     # Node.js + Express (API + sync)
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── handler.ts           # Lambda entry point (wraps Express app)
│   │   ├── app.ts               # Express app setup
│   │   ├── routes/
│   │   │   ├── auth.ts          # Better Auth routes
│   │   │   ├── sync.ts          # POST /sync, GET /sync/status
│   │   │   ├── horses.ts        # GET /horses, GET /horses/:id, PATCH exclude
│   │   │   ├── herds.ts         # GET /herds, GET /herds/:id/horses
│   │   │   ├── photos.ts        # GET /photos/:id, PATCH exclude/include
│   │   │   └── identify.ts      # POST /identify (upload → invoke ML worker → pgvector query)
│   │   ├── db/
│   │   │   ├── schema.ts        # Drizzle schema definitions
│   │   │   ├── client.ts        # Neon serverless driver connection
│   │   │   └── migrate.ts       # Migration runner
│   │   ├── services/
│   │   │   ├── drive.ts         # Google Drive API: list folders, list files, diff
│   │   │   ├── sync.ts          # Sync orchestration: diff → DB updates → fan-out workers
│   │   │   ├── config.ts        # SSM Parameter Store loader (+ .env fallback)
│   │   │   └── websocket.ts     # Push progress to WebSocket clients
│   │   └── auth/
│   │       └── index.ts         # Better Auth server setup
│   └── Dockerfile               # Node.js Lambda image
│
├── workers/                     # Python ML workers
│   ├── requirements.txt
│   ├── Dockerfile               # ML image (torch, timm, YOLO, wildlife-tools, model pre-downloaded)
│   ├── handler.py               # Lambda entry: routes to detector or extractor based on event
│   ├── detector.py              # Multi-horse detection (reuses horse_detection_lib.py logic)
│   ├── extractor.py             # Feature extraction (reuses wildlife-mega-L-384 pipeline)
│   ├── identifier.py            # Query image identification (extract features → pgvector query)
│   ├── drive_client.py          # Download images from Drive by file ID
│   └── db.py                    # psycopg2 + pgvector writes
│
├── sms/                         # Twilio SMS handlers (updated from existing)
│   ├── Dockerfile               # Lightweight: boto3 + twilio only
│   ├── webhook_responder.py     # Receives Twilio webhook, invokes identifier worker
│   └── requirements.txt
│
├── db/
│   └── migrations/
│       └── 001_initial.sql      # Schema + pgvector extension
│
├── scripts/
│   ├── migrate_to_drive.py      # One-time: build Drive folder structure from existing data
│   └── seed_ssm.sh              # Populate SSM parameters from .env
│
└── tests/
    ├── backend/                 # Jest tests for API routes
    └── workers/                 # Pytest for ML workers
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
    processing_status TEXT DEFAULT 'pending',  -- pending, detecting, extracting, ready, excluded
    detection_result TEXT,                      -- NULL, NONE, SINGLE, MULTIPLE
    excluded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

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
    files_scanned INTEGER DEFAULT 0,
    files_added INTEGER DEFAULT 0,
    files_removed INTEGER DEFAULT 0,
    files_moved INTEGER DEFAULT 0
);

-- Better Auth manages its own user/session tables automatically
```

## Lambda Functions (4 container images)

| Function | Runtime | Image | Memory | Timeout | Trigger |
|----------|---------|-------|--------|---------|---------|
| **api** | Node.js | `backend/` | 512 MB | 30s | API Gateway HTTP |
| **ml-worker** | Python | `workers/` | 4096 MB | 15 min | Async invoke |
| **webhook-responder** | Python | `sms/` | 256 MB | 10s | API Gateway HTTP |
| **websocket-handler** | Node.js | `backend/` (same image) | 256 MB | 10s | API Gateway WebSocket |

The **ml-worker** handles detection, extraction, AND identification — the handler inspects `event.task` to route to the right function. One fat container image with both models pre-downloaded.

The **webhook-responder** stays Python (minimal changes from existing). It invokes the ml-worker for identification instead of the old horse-id-processor.

## Implementation Phases

### Phase 1: Migration to Drive
**Populate Google Drive with existing data so all subsequent phases have real data to work with.**

1. Set up Google Drive service account + share root folder
2. `cloud/scripts/migrate_to_drive.py`:
   - Read existing `horse_herds.csv` + merged manifest CSV
   - Create top-level folder per herd in Drive
   - Create horse subfolder per horse within each herd
   - Upload photos from S3/local storage into the appropriate folders
   - Log mapping of Drive folder/file IDs for verification
3. Verify: folder structure matches expected herds/horses/photo counts
4. This is a standalone script — no cloud infrastructure needed yet, just `google-api-python-client` + existing data files

**Key files to read:**
- `horse_herds.csv` — herd→horse mapping
- Merged manifest CSV — photo→horse mapping + filenames
- S3 bucket `horse-id-data` — source photos

### Phase 2: Foundation
**Database + Drive sync + local dev environment. Tested against real Drive data from Phase 1.**

1. Create Neon project, enable pgvector, run schema migration
2. `cloud/db/migrations/001_initial.sql` — schema above
3. `cloud/backend/src/db/` — Drizzle schema + Neon serverless driver
4. `cloud/backend/src/services/drive.ts` — Google Drive API client:
   - `listHerdFolders(rootFolderId)` → top-level folders
   - `listHorseFolders(herdFolderId)` → second-level folders
   - `listImageFiles(horseFolderId)` → image files with metadata
   - `getFileMetadata(fileId)` → md5Checksum, name, parents
5. `cloud/backend/src/services/sync.ts` — sync orchestration:
   - Walk Drive tree, diff against DB
   - Apply changes: create/update/delete herds, horses, photos
   - Queue new photos for processing (invoke ml-worker in batches)
   - Track progress in sync_runs table
6. `cloud/docker-compose.yml` — Postgres+pgvector, Node.js backend
7. `cloud/.env.example` — template for local config
8. Run initial sync against real Drive → verify DB matches expected data

### Phase 3: ML Workers
**Detection + extraction running in Lambda, writing to pgvector.**

1. `cloud/workers/Dockerfile` — based on existing `Dockerfile.horse_id`:
   - Python 3.13 Lambda base
   - Pre-download YOLO model + wildlife-mega-L-384
   - System deps from existing (mesa-libGL, etc.)
2. `cloud/workers/handler.py` — routes `event.task` to detector/extractor/identifier
3. `cloud/workers/detector.py`:
   - Receives batch of photo IDs
   - Downloads images from Drive via `drive_client.py`
   - Runs YOLO classification (port logic from `horse_detection_lib.py`)
   - Writes detection_result + processing_status to DB
4. `cloud/workers/extractor.py`:
   - Receives batch of photo IDs (SINGLE only)
   - Downloads images from Drive
   - Extracts embeddings via `DeepFeatures(backbone)` (same pattern as `horse_id.py:322`)
   - INSERTs vectors into features table via pgvector
   - Updates processing_status to 'ready'
5. `cloud/workers/identifier.py`:
   - Receives image URL/bytes
   - Extracts query embedding
   - Runs pgvector similarity query
   - Returns top-N results
6. `cloud/workers/db.py` — psycopg2 connection + pgvector registration

**Reused from existing codebase:**
- `horse_detection_lib.py` → `classify_horse_detection()`, depth analysis, edge cropping logic
- `Dockerfile.horse_id` → base image, system deps, model pre-download pattern
- `horse_id.py` lines 312-325 → model loading, transform pipeline, DeepFeatures usage
- `config.yml` detection section → thresholds and parameters

### Phase 4: API Routes
**Express API serving data and triggering operations.**

1. `cloud/backend/src/app.ts` — Express setup + middleware
2. `cloud/backend/src/handler.ts` — Lambda wrapper (serverless-http or @vendia/serverless-express)
3. Routes:
   - `POST /api/sync` → triggers sync (invokes sync logic, returns sync_run ID)
   - `GET /api/sync/:id` → sync status
   - `GET /api/herds` → list herds with horse counts
   - `GET /api/herds/:id/horses` → horses in herd with photo counts
   - `GET /api/horses/:id` → horse detail with photos
   - `PATCH /api/photos/:id` → toggle exclude/include
   - `POST /api/identify` → upload image, optional herd filter, returns matches
   - `GET /api/photos/:id/image` → proxy image from Drive (avoids exposing Drive creds to client)
4. `cloud/backend/src/services/config.ts` — SSM loader with .env fallback

**Identification flow (POST /api/identify):**
1. Accept multipart image upload
2. Save to /tmp, invoke ml-worker with `task: 'identify'`
3. Worker extracts features, queries pgvector, returns results
4. API returns ranked matches with horse names, herds, confidence, reference photo IDs

### Phase 5: SMS Flow Update
**Existing Twilio SMS backed by Neon.**

1. `cloud/sms/webhook_responder.py` — port from existing `webhook_responder.py`:
   - Same Twilio signature validation
   - Parse herd filter from message text
   - Invoke ml-worker (instead of old horse-id-processor) with `task: 'identify'`
   - Config from SSM/env vars
2. ML worker's `identifier.py` handles the actual identification (shared with web flow)
3. Worker sends results back via Twilio API (same as current `horse_id.py:474-483`)

### Phase 6: WebSocket Progress
**Real-time updates during sync and processing.**

1. API Gateway WebSocket API with $connect/$disconnect/$default routes
2. `cloud/backend/src/services/websocket.ts`:
   - Track connection IDs in memory (or simple DynamoDB table if needed)
   - `broadcast(event)` → POST to API Gateway Management API
3. Sync and workers post progress events: `{type: 'sync_progress', filesScanned, ...}`, `{type: 'photo_status', photoId, status}`
4. Workers call a shared "notify" function (invoke a small Lambda or write to DynamoDB stream)

### Phase 7: Frontend
**React SPA.**

1. **Pages:**
   - `/login`, `/signup` — Better Auth client
   - `/` — Dashboard: sync button, last sync info, processing queue
   - `/herds` → `/herds/:id` — herd list → horses in herd
   - `/horses/:id` — horse detail with photo grid
   - `/identify` — upload/camera capture, herd filter dropdown, results
2. **Components:**
   - `SyncButton` + `SyncProgress` (WebSocket-driven)
   - `HorseCard` (thumbnail, name, herd, photo count)
   - `PhotoGrid` (with exclude toggle per photo, status indicators)
   - `IdentifyForm` (drag-drop/camera, herd select)
   - `MatchResults` (ranked cards with confidence bars + reference photos)
3. Photo display: all images proxied via `GET /api/photos/:id/image`
4. Better Auth React client for auth state

### Phase 8: IaC + Deployment
**SAM template + CI/CD.**

1. `cloud/template.yaml` — SAM template:
   - API Gateway HTTP API
   - API Gateway WebSocket API
   - Lambda functions (4)
   - S3 bucket for frontend
   - CloudFront distribution
   - SSM parameter references
   - IAM roles (Lambda execution, Drive access, SSM read)
2. Update `buildspec.yml` — build Docker images + frontend, `sam deploy`
3. `cloud/scripts/seed_ssm.sh` — populate SSM from .env for initial deploy

### Phase 9: Production Cutover
**Switch live traffic to new system.**

1. Deploy full stack via SAM
2. Trigger production sync (Drive already populated in Phase 1)
3. Verify: horse counts, photo counts, feature vector counts match expected
4. Run identification on known test images, compare accuracy to current system
5. Point Twilio webhook URL to new webhook-responder
6. Deprecate old S3 CSVs/pickles

### Phase 10: Legacy Cleanup
**Move old CLI/CSV pipeline code out of the way.**

Move the following to a `legacy/` folder (preserving git history):
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
- `Dockerfile.horse_id`, `Dockerfile.responder` — old Lambda images (replaced by `cloud/` images)
- `horse-id-requirements.txt`, `responder-requirements.txt` — old requirements
- Old `tests/` that reference removed scripts

Keep in project root (still used):
- `horse_detection_lib.py` — detection logic reused by ml-worker (or copy into `cloud/workers/`)
- `config.yml` — reference for detection thresholds (parameters migrated to SSM)
- `horse_id.py`, `webhook_responder.py` — reference only (logic ported to `cloud/`)
- `CLAUDE.md` — update to reflect new architecture

## Testing Strategy

Essential integration tests only — focused on the critical paths that would be painful to debug without tests.

**~15-20 tests across 4 test files:**

### `cloud/tests/backend/sync.test.ts` (Jest)
- Sync with mocked Drive API → verify correct herds/horses/photos created in DB
- Sync detects new images, moved horses, deleted images
- Sync skips unchanged images (matching drive_file_id + md5)
- Sync handles renamed folders (herd or horse rename in Drive)

### `cloud/tests/workers/test_detector.py` (Pytest)
- Detection classifies known SINGLE/MULTIPLE/NONE images correctly
- Detection writes results to DB and updates processing_status
- Batch of mixed images processed correctly
- Port relevant cases from existing `tests/test_detection_algorithms.py`

### `cloud/tests/workers/test_identifier.py` (Pytest)
- Feature extraction produces 384-dim vector, written to pgvector
- Identification query returns correct top-N matches
- Herd filter narrows results correctly
- End-to-end: image in → correct horse name out

### `cloud/tests/backend/identify.test.ts` (Jest)
- POST /api/identify with valid image returns ranked matches
- POST /api/identify with herd filter returns only horses from that herd
- SMS flow: webhook → worker → Twilio response (mocked Twilio)

**Infrastructure:**
- Tests run against Docker Compose Postgres+pgvector (real DB, not mocked)
- Drive API mocked in sync tests (return canned folder/file listings)
- ML model calls are real in worker tests (need the model available)
- `npm test` / `pytest` from `cloud/` directory

**What's NOT tested:**
- Individual Express route handlers (trust the framework)
- Frontend components (manual QA)
- Better Auth flows (trust the library)
- Google Drive API itself (mocked)

## Verification

**Local:**
- `docker-compose up` → Postgres+pgvector + backend running
- Create test Drive folder with 2 herds, 3 horses, ~10 photos
- Trigger sync, verify DB populated correctly
- Run detection + extraction on test photos
- POST /api/identify with test image, verify correct match
- Connect WebSocket, verify progress events arrive
- `npm test` and `pytest` pass

**Production:**
- `sam deploy`, verify all Lambdas healthy
- Run migration, trigger sync
- Browse herds/horses in web UI
- Identify a known horse via web UI
- Send test SMS, verify response
- Compare top-5 accuracy with current system on standard test set
