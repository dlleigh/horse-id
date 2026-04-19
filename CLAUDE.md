# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Horse Identity Matching System — identifies individual horses from photos using computer vision. Photos are sourced from Google Drive, processed by AWS Lambda ML workers (YOLO detection + Wildlife-mega-L-384 embeddings), stored in Neon Postgres with pgvector, and served via a React SPA. SMS/MMS identification is handled via Twilio.

## Project Structure

```
├── cloud/
│   ├── frontend/          # React + Vite + TypeScript SPA
│   ├── backend/           # Node.js + Express API server (Drizzle ORM)
│   ├── workers/           # Python ML Lambda (detect, extract, identify, sync_batch)
│   ├── db/migrations/     # Postgres migration SQL files
│   └── scripts/           # One-time migration scripts
├── webhook_responder.py   # Twilio SMS Lambda handler
├── Dockerfile.responder   # SMS responder container image
├── horse_detection_lib.py # Detection logic (shared with ML worker)
├── config_utils.py        # Config loader (used by horse_detection_lib)
├── config.yml             # Detection thresholds (copied into ML worker image)
├── buildspec.yml          # CodeBuild CI/CD pipeline
├── responder-requirements.txt
└── legacy/                # Old CLI/CSV pipeline (moved here for reference)
```

## Key Commands

### Backend Development
```bash
cd cloud/backend && npm run dev     # Start Express API server
cd cloud/backend && npx tsx src/db/migrate.ts  # Run DB migrations
```

### Frontend Development
```bash
cd cloud/frontend && npm run dev    # Start Vite dev server
cd cloud/frontend && npm run build  # Production build
```

### Docker Images
```bash
# ML worker (Lambda container)
docker build --platform linux/amd64 -f cloud/workers/Dockerfile -t horse-id-ml-worker .

# Twilio webhook responder (Lambda container)
docker build --platform linux/amd64 -f Dockerfile.responder -t horse-id-responder .
```

### Deployment
CodeBuild runs automatically from `buildspec.yml`: builds both Docker images, pushes to ECR, and updates Lambda functions.

## Architecture

### Event-Driven Pipeline
```
Sync button → Backend calls Drive Changes API → batches → ml-worker (sync_batch)
  → ml-worker (detect) → ml-worker (extract) → photo "ready" in pgvector
```

Each Lambda invocation chains to the next. No orchestrator needed.

### Lambda Functions
| Function | Runtime | Purpose |
|----------|---------|---------|
| **horse-id-ml-worker** | Python | Detection, extraction, identification, sync batches |
| **twilio-webhook-responder** | Python | Receives Twilio webhooks, invokes ml-worker for ID |

### Key Technologies
- **Wildlife-mega-L-384** — feature extraction for horse re-identification
- **YOLO** — multi-horse detection (NONE/SINGLE/MULTIPLE classification)
- **Neon Postgres + pgvector** — database with vector similarity search
- **Google Drive** — photo source of truth (Changes API for incremental sync)
- **Drizzle ORM** — TypeScript-native database access
- **Twilio** — SMS/MMS interface

### Recovery
`POST /api/process` resets stuck photos (>15 min in detecting/extracting) and fans out pending work to Lambda.

## Configuration

- `config.yml` — detection thresholds, YOLO model config
- `cloud/backend/.env` — DATABASE_URL, Google Drive service account, AWS config
- Lambda env vars: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `PROCESSOR_LAMBDA_NAME`
- Secrets stored in AWS SSM Parameter Store

## Development Notes

- Backend runs as standalone Express server (not behind API Gateway)
- Frontend uses 2-second HTTP polling for live updates (WebSocket not implemented)
- No auth yet (Better Auth deferred)
- `horse_detection_lib.py` and `config.yml` live at project root because they're COPY'd into the ML worker Docker image at build time

### Before Running Python Commands

```bash
# ALWAYS activate venv first
source venv/bin/activate
```
