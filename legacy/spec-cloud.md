# Horse ID Cloud Platform — Specification

## Overview

Migrate the Horse Identity Matching System from a local CLI/email-based pipeline to a browser-based web application with cloud backend. Users organize horse photos in Google Drive using a folder hierarchy (herd → horse → photos), and the system syncs from Drive on demand. Multi-horse detection and feature extraction run in the background. The SMS/Twilio identification interface will continue to work, updated to share the same backend as the web app.

## Current State → Target State

| Aspect | Current | Target |
|--------|---------|--------|
| Photo ingestion | Email (Gmail API) | Google Drive sync |
| Name normalization | Interactive CLI | Derived from folder names |
| Horse detection | Local script | Background cloud job |
| Feature extraction | Local script | Background cloud job |
| Data storage | CSV files on disk | Database |
| Photo storage | Local disk + S3 | Google Drive (+ S3 cache) |
| Identification | SMS/Twilio only | Web app + SMS/Twilio |
| Auth | None | Multi-user authentication |
| Gallery/browsing | Static HTML + Streamlit | Web UI |

## Architecture

### Frontend

Single-page web application. No framework preference — choose what best fits the use case (React, Next.js, etc.). Key requirement: real-time progress feedback during background processing via WebSockets or SSE.

### Backend API (Lambda)

API Gateway + Lambda functions handling:
- Authentication and session management
- Google Drive sync trigger and status
- Horse/herd browsing and management
- Horse identification queries
- Background job status and progress

### Background Workers (Lambda)

Separate Lambda functions for compute-heavy tasks:
- **Multi-horse detection** (YOLOv8-Seg) — classifies images as NONE/SINGLE/MULTIPLE
- **Feature extraction** (Wildlife-mega-L-384) — extracts 384-dim embeddings for similarity matching
- Run on CPU (no GPU required), invoked asynchronously
- **Fan-out pattern:** Sync identifies new/changed images, then invokes worker Lambdas in batches. Batch size is configurable (to amortize model load cold start vs. staying within the 15-minute Lambda timeout). Each worker processes its batch and writes results to the database.
- Must push real-time progress updates to the frontend

### Database

Neon (serverless Postgres) with pgvector extension. Scales to zero when idle — no cost during inactive periods. Free tier provides 0.5 GB storage, 100 compute-hours/month, which is sufficient for this workload.

Schema must support:
- Horses (canonical_id, name, herd, status)
- Photos (filename, drive_file_id, horse reference, drive metadata/hash, detection result, exclusion status)
- Features (horse reference, embedding vector via pgvector, extraction timestamp)
- Herds (name, derived from top-level Drive folders)
- Users (auth credentials, profile)
- Sync state (last sync timestamp, per-file change tracking)

**Data integrity rule preserved:** All photos with the same canonical_id MUST have the same horse name.

**Naming convention:** Horse names and herd names are derived directly from Drive folder names. Within a given herd, all horse folder names must be unique (enforced by the user in Drive). The same horse name may appear in different herds — these are treated as distinct horses.

### Photo Storage

Google Drive is the primary photo storage. The system reads photos from Drive during sync. Optionally, photos may be cached or copied to S3 for faster access by the backend and Lambda functions.

### SMS/Twilio Integration

Update the existing Lambda-based SMS flow to query the new database and feature store instead of S3 CSVs. The webhook-responder and horse-id-processor Lambdas will be updated to:
- Read the manifest/features from the database instead of S3 CSV/pickle files
- Share the same source of truth as the web app

## Features

### 1. Authentication

- Multi-user authentication (sign-up, login, logout)
- Use Neon Auth (built on Better Auth) — stores users/sessions directly in the Neon database, no external auth provider needed
- All authenticated users have equal permissions (no role-based access control)
- Single-tenant: all users share one pool of horses

### 2. Google Drive Sync

Users organize photos in a shared Google Drive folder with this structure:

```
<root folder>/
  <Herd A>/
    <Horse 1>/
      photo1.jpg
      photo2.jpg
    <Horse 2>/
      photo1.jpg
  <Herd B>/
    <Horse 1>/      ← same name as in Herd A, treated as a different horse
      photo1.jpg
```

**Sync behavior:**
- Users trigger a sync from the web UI (no automatic polling)
- The system scans the Drive folder tree and compares against its database
- **New images:** Added to the database, background processing kicks off automatically
- **Moved images** (horse moved to different herd folder): The horse's herd assignment is updated in the database
- **Deleted images:** Removed from the database and matching index
- **Unchanged images:** Skipped entirely (tracked via Drive file ID and content hash)
- **New herd/horse folders:** Created automatically in the database
- **Empty folders** (after moves/deletes): Cleaned up in the database

**Horse identity from folder structure:**
- Herd name = top-level folder name
- Horse name = second-level folder name
- A horse is uniquely identified by (herd, name) pair
- The user is responsible for keeping horse names unique within a herd in Drive
- Renaming a folder in Drive results in a rename in the database on next sync

**Progress:** Real-time sync progress is pushed to the frontend (files scanned, changes detected, processing queued)

### 3. Background Processing Pipeline

After sync discovers new or updated images:
1. **Multi-horse detection** runs on each new image
   - Classifies as NONE / SINGLE / MULTIPLE
   - Results stored in database
   - Real-time progress pushed to frontend
2. **Feature extraction** runs on SINGLE-horse images
   - Extracts Wildlife-mega-L-384 embeddings
   - Features stored in database
   - Real-time progress pushed to frontend

Users see processing status per-photo in the UI (pending → detecting → extracting → ready, or → excluded if NONE/MULTIPLE).

### 4. Horse & Photo Management

- Browse all horses, view their photos
- Filter by herd
- Exclude/include individual photos (mark bad photos, crop issues, etc.)
- Excluding a photo removes it from the matching database; re-including it triggers feature re-extraction
- Photo management (adds, deletes, moves, renames) happens in Google Drive; the system reflects changes on next sync

### 5. Herd Management

- Herds are derived from top-level Drive folders — created/renamed/deleted in Drive, reflected on sync
- View horses by herd in the web UI
- Reassigning a horse to a different herd = moving its folder in Drive

### 6. Horse Identification (Web)

- Upload or take a photo to identify a horse
- Optionally filter by herd to narrow results
- Returns top-N matches with confidence scores and reference photos
- Uses the same similarity engine as the SMS flow (cosine similarity on Wildlife-mega-L-384 embeddings)

### 7. Horse Identification (SMS)

- Existing Twilio SMS/MMS interface continues to work
- Updated to read from the new database instead of S3 CSVs
- Same identification logic and thresholds

## Non-Goals (Out of Scope)

- GPU-based processing
- Multi-tenant / organization isolation
- Role-based access control
- Custom domain
- Mobile-native app (web app should be mobile-friendly though)
- Automated re-training or model fine-tuning
- Email ingestion (fully replaced by Google Drive sync)
- Browser-based photo upload (all photo management happens in Google Drive)
- Automatic/scheduled Drive sync (user triggers manually)

## Migration

Migration populates both Google Drive and the database from existing data:

1. **Build Google Drive folder structure** from existing herds and horses:
   - Create top-level folder per herd
   - Create horse subfolder per horse within each herd
   - Copy photos from S3/local storage into the appropriate horse folders
   - This establishes Google Drive as the source of truth going forward

2. **Run initial sync** against the newly populated Drive to seed the database (horses, photos, herds, sync state)

3. **Re-extract features** into pgvector (or migrate existing embeddings if compatible)

4. **Update SMS flow** to use new backend; old S3-based CSVs/pickles deprecated after migration

## Local Development

Full stack runs locally via Docker Compose:
- Frontend, backend API, and background workers in containers
- Postgres + pgvector container (drop-in for Neon)
- Google Drive: points at production Drive (read-only access, safe to share between local and cloud environments)
- docker-compose.yml provides the complete local stack

## Infrastructure

- Compute: AWS Lambda for all backend and worker functions (serverless, scales to zero)
- API: API Gateway + Lambda
- Database: Neon serverless Postgres (with pgvector for embeddings, Neon Auth for authentication)
- Photo storage: Google Drive (primary), S3 (optional cache for Lambda access)
- Google Drive API: service account or OAuth for Drive access
- Frontend hosting: S3 + CloudFront (static SPA) or equivalent
- Configuration & secrets: AWS SSM Parameter Store for all config and secrets (SecureString for credentials). Includes Neon connection string, Google Drive service account key, Twilio credentials, worker batch size, Drive root folder ID, etc.
- Local dev: `.env` file (not committed) as the local equivalent
- CI/CD: existing CodeBuild pipeline can be extended or replaced
