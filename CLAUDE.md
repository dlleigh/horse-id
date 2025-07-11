# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Horse Identity Matching System that identifies individual horses based on photos. The system processes emails containing horse photos, detects horses, merges photos of the same horse from different sources, and provides SMS/MMS identification via Twilio.

## Key Commands

### Core Processing Pipeline
```bash
# 1. Ingest emails and extract horse photos
python ingest_from_email.py

# 2. Detect number of horses in each photo
python multi_horse_detector.py

# 3. Merge identities of horses with same name
python merge_horse_identities.py

# 4. Review merge decisions interactively
streamlit run review_merges_app.py

# 5. Generate HTML galleries for visual review
python generate_gallery.py

# 6. Extract features for similarity matching
python extract_features.py

# 7. Upload data to S3
python upload_to_s3.py
```

### Testing and Development
```bash
# Test Lambda functions locally
streamlit run test_lambda_app.py

# Run Jupyter notebook for calibration and testing
jupyter notebook horse_id_ensemble.ipynb
```

### Docker Deployment
```bash
# Build processor image
docker build --platform linux/amd64 -f Dockerfile.horse_id -t horse-id-processor .

# Build responder image
docker build --platform linux/amd64 -f Dockerfile.responder -t horse-id-responder .
```

## System Architecture

### Data Flow Pipeline
1. **Email Ingestion** (`ingest_from_email.py`) - Fetches emails, extracts horse names, saves images
2. **Multi-Horse Detection** (`multi_horse_detector.py`) - Uses YOLO to classify images as NONE/SINGLE/MULTIPLE horses
3. **Identity Merging** (`merge_horse_identities.py`) - Uses WildFusion similarity to merge photos of same horse
4. **Merge Review** (`review_merges_app.py`) - Interactive web app for correcting merge decisions
5. **Feature Extraction** (`extract_features.py`) - Pre-extracts features using Wildlife-mega-L-384 model
6. **AWS Lambda Deployment** - Two-function architecture for real-time horse identification

### Key Technologies
- **WildlifeTools Framework** - Core similarity matching and feature extraction
- **YOLOv5** - Multi-horse detection
- **Streamlit** - Web interfaces for review and testing
- **AWS Lambda** - Production deployment with dual-function architecture
- **Twilio** - SMS/MMS interface

### CSV Data Files
- `manifest_file` - Initial photos from email ingestion
- `detected_manifest_file` - After horse detection analysis
- `merged_manifest_file` - Final merged identities
- `merge_results_file` - Log of similarity comparisons

### Lambda Architecture
- **webhook-responder** - Receives Twilio webhooks, returns immediate response
- **horse-id-processor** - Performs actual horse identification asynchronously

## Configuration

- `config.yml` - Central configuration for all paths, thresholds, and settings
- `credentials.json` + `token.json` - Gmail API authentication
- Requires AWS CLI configured with appropriate permissions for S3 and Lambda
- Environment variables needed for Lambda: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `PROCESSOR_LAMBDA_NAME`

## Development Notes

- The system uses a specific workflow order - email ingestion must happen before detection, detection before merging, etc.
- Calibration files (`.pkl`) are required for similarity matching and are created by the Jupyter notebook
- The system is designed to handle incremental processing - scripts can be re-run to process new data
- All file paths use `{data_root}` placeholder pattern for environment-agnostic configuration