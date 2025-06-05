import os
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import re
from datetime import datetime
from email.message import EmailMessage
from dateutil import parser, tz
import csv
import torch # Though not directly used, wildlife-tools might expect it
import torchvision.transforms as T
import timm
from wildlife_tools.features import (
    DeepFeatures, SuperPointExtractor, AlikedExtractor, DiskExtractor, SiftExtractor
)
from wildlife_tools.similarity import CosineSimilarity, MatchLOFTR, MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.data import ImageDataset # Make sure this is the correct import if used
from PIL import Image
import pandas as pd
import numpy as np
import pickle

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'

DATA_ROOT = os.environ.get('HORSE_ID_DATA_ROOT', '%s/google-drive/horseID Project/data' % os.environ.get('HOME', '.'))
DATASET_DIR = '%s/horse_photos' % DATA_ROOT
MANIFEST_FILE = '%s/horse_photos_manifest.csv' % DATA_ROOT
CALIBRATION_DIR = '%s/calibrations' % DATA_ROOT # For loading .pkl calibration files
TEMP_DIR = '%s/tmp' % DATA_ROOT

REPLACE_EXISTING = False
SIMILARITY_THRESHOLD = 0.9  # ! IMPORTANT: Tune this threshold based on validation
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']

# --- Global variables for the WildFusion system ---
parser_wildfusion_system = None
parser_priority_matcher = None
parser_calibrated_matchers_dict = None
# --- End Global variables ---

def gmail_authenticate():
    """Shows basic usage of the Gmail API.
    Performs authentication and returns the Gmail API service object.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"Error loading token from {TOKEN_FILE}: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("Credentials expired. Refreshing token...")
                creds.refresh(Request())
                print("Token refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing token: {e}")
                creds = None
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"ERROR: Credentials file '{CREDENTIALS_FILE}' not found.")
                print("Please download it from Google Cloud Console and place it in the same directory.")
                return None
            try:
                print("No valid token found. Starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=8080)
                print("OAuth flow completed. Credentials obtained.")
            except Exception as e:
                print(f"Error during OAuth flow: {e}")
                return None
        try:
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
            print(f"Token saved to {TOKEN_FILE}")
        except Exception as e:
            print(f"Error saving token: {e}")

    if creds and creds.valid:
        print("Authentication successful.")
        try:
            service = build('gmail', 'v1', credentials=creds)
            return service
        except Exception as e:
            print(f"Error building Gmail service: {e}")
            return None
    else:
        print("Failed to obtain valid credentials.")
        return None

def get_horse_emails(service):
    """Fetch all emails"""
    try:
        results = service.users().messages().list(userId='me', q='').execute()
        messages = results.get('messages', [])
        return messages
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def extract_horse_name(subject):
    """Extract horse name from email subject"""
    subject = re.sub(r'^Fwd?:\s*', '', subject, flags=re.IGNORECASE)
    match = re.match(r'^([^-]+)-?\s*(?:fall|spring|summer|winter).*$', subject, re.IGNORECASE)
    if match:
        horse_name = match.group(1)
        horse_name = re.sub(r'[^\w\s-]', '', horse_name).strip()
        return horse_name
    return None

def extract_oldest_date(message_gmail_api): # Renamed parameter for clarity
    """Extract the oldest sent date from email headers and forwarded content"""
    dates = []
    tzinfos = {"CDT": tz.gettz("America/Chicago"), "CST": tz.gettz("America/Chicago"),
                 "EDT": tz.gettz("America/New_York"), "EST": tz.gettz("America/New_York"),
                 "PDT": tz.gettz("America/Los_Angeles"), "PST": tz.gettz("America/Los_Angeles"),
                 "MDT": tz.gettz("America/Denver"), "MST": tz.gettz("America/Denver")}

    def process_part_for_date(part):
        if 'parts' in part:
            for subpart in part['parts']:
                process_part_for_date(subpart)
            return
        if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
            try:
                body_bytes = base64.urlsafe_b64decode(part['body']['data'])
                body_text = body_bytes.decode('utf-8', errors='replace')
                header_patterns = [r'(?:Sent|Date|From):.+$', r'On.+wrote:$', r'>.+\d{4}.*$']
                for line in body_text.split('\n'):
                    line_stripped = line.strip()
                    if any(re.search(pattern, line_stripped, re.IGNORECASE) for pattern in header_patterns):
                        try:
                            parsed_date = parser.parse(line_stripped, fuzzy=True, tzinfos=tzinfos)
                            date_only = parsed_date.date()
                            if date_only.year > 1970:
                                dates.append(date_only)
                        except (ValueError, parser.ParserError, OverflowError):
                            continue
            except Exception as e:
                print(f"Warning: Error decoding or processing part for date: {e}")


    payload = message_gmail_api.get('payload', {})
    process_part_for_date(payload)
    
    try:
        internal_timestamp = int(message_gmail_api.get('internalDate', 0)) / 1000
        if internal_timestamp > 0 : # Check if timestamp is valid
            internal_date_obj = datetime.fromtimestamp(internal_timestamp).date()
            dates.append(internal_date_obj)
        else: # Fallback if internalDate is 0 or invalid
            print(f"Warning: Invalid internalDate for message. Using current date as fallback for oldest_date logic.")
            dates.append(datetime.now().date())

    except ValueError: # Handle potential errors converting internalDate
        print(f"Warning: Could not parse internalDate. Using current date as fallback for oldest_date logic.")
        dates.append(datetime.now().date())


    oldest_date_obj = min(dates) if dates else datetime.now().date()
    return oldest_date_obj.strftime('%Y%m%d')

def load_production_system_for_parser():
    """
    Loads the pre-trained WildFusion system components for use within the parser.
    This function should be called once.
    """
    global parser_wildfusion_system, parser_priority_matcher, parser_calibrated_matchers_dict

    if parser_wildfusion_system is not None:
        print("Parser's WildFusion system components already loaded.")
        return

    print("Loading WildFusion system components for parser...")

    parser_calibrated_matchers_dict = {
        'lightglue_superpoint': SimilarityPipeline(
            matcher=MatchLightGlue(features='superpoint'), extractor=SuperPointExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
        'lightglue_aliked': SimilarityPipeline(
            matcher=MatchLightGlue(features='aliked'), extractor=AlikedExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
        'lightglue_disk': SimilarityPipeline(
            matcher=MatchLightGlue(features='disk'), extractor=DiskExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
        'lightglue_sift': SimilarityPipeline(
            matcher=MatchLightGlue(features='sift'), extractor=SiftExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
        # Add other SimilarityPipeline definitions from your ensemble if they were saved
        # Example for MegaDescriptor if it was part of the calibrated ensemble:
        # 'wildlife_mega_L_384': SimilarityPipeline(
        #     matcher=CosineSimilarity(),
        #     extractor=DeepFeatures(
        #         model=timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)
        #     ),
        #     transform=T.Compose([
        #         T.Resize(size=(384, 384)), T.ToTensor(),
        #         T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ]),
        #     calibration=IsotonicCalibration()
        # ),
    }

    all_loaded_successfully = True
    for name, pipeline in parser_calibrated_matchers_dict.items():
        cal_file = os.path.join(CALIBRATION_DIR, f"{name}.pkl")
        if os.path.exists(cal_file):
            print(f"Loading calibration for {name} from: {cal_file}")
            try:
                with open(cal_file, 'rb') as f:
                    pipeline.calibration = pickle.load(f)
                # Assuming IsotonicCalibration's load method makes it ready
                # If SimilarityPipeline itself needs a flag:
                if hasattr(pipeline, 'calibration_done'):
                    pipeline.calibration_done = True
            except Exception as e:
                print(f"Error loading calibration {cal_file}: {e}")
                if hasattr(pipeline, 'calibration_done'):
                    pipeline.calibration_done = False
                all_loaded_successfully = False
        else:
            print(f"Calibration file not found for {name}: {cal_file}. This matcher will be uncalibrated.")
            if hasattr(pipeline, 'calibration_done'):
                pipeline.calibration_done = False
            all_loaded_successfully = False

    if not all_loaded_successfully:
        print("WARNING: Not all matcher calibrations were loaded. Similarity checks might be suboptimal or fail if uncalibrated pipelines are not handled.")
        # Consider raising an error or having a fallback if essential calibrations are missing

    parser_priority_matcher = SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(
            model=timm.create_model(
                'hf-hub:BVRA/wildlife-mega-L-384',
                num_classes=0,
                pretrained=True
            )
        ),
        transform=T.Compose([
            T.Resize(size=(384, 384)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
    )

    # Use 'calibrated_pipelines' and 'priority_pipeline' as per user's notebook context for WildFusion
    parser_wildfusion_system = WildFusion(
        calibrated_pipelines=list(parser_calibrated_matchers_dict.values()),
        priority_pipeline=parser_priority_matcher
    )
    print("Parser's WildFusion system components loaded.")


def check_horse_similarity(new_image_paths: list, existing_horse_image_paths: list, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """
    Check if a set of new images likely matches an existing horse using the WildFusion ensemble.
    """
    global parser_wildfusion_system
    if parser_wildfusion_system is None:
        print("WildFusion system for parser not loaded. Attempting to load...")
        try:
            load_production_system_for_parser()
            if parser_wildfusion_system is None:
                raise SystemError("Failed to load WildFusion system for parser.")
        except Exception as e:
            print(f"Fatal: Could not load WildFusion system: {e}")
            return False

    if not new_image_paths or not existing_horse_image_paths:
        print("Warning: Empty image list provided for similarity check (new or existing).")
        return False

    df_query_list = [{'image_id': f"new_q_{i}", 'identity': 'query_horse', 'path': os.path.abspath(p), 'date': pd.Timestamp.now()}
                     for i, p in enumerate(new_image_paths) if os.path.exists(os.path.abspath(p))]
    if not df_query_list:
        print("Warning: No valid new image paths found for query.")
        return False
    df_query = pd.DataFrame(df_query_list)
    dataset_query = ImageDataset(df_query, root='/')

    df_db_list = [{'image_id': f"db_exist_{i}", 'identity': 'existing_horse', 'path': os.path.abspath(p), 'date': pd.Timestamp.now()}
                  for i, p in enumerate(existing_horse_image_paths) if os.path.exists(os.path.abspath(p))]
    if not df_db_list:
        print("Warning: No valid existing horse image paths found for database.")
        return False
    df_database = pd.DataFrame(df_db_list)
    dataset_database = ImageDataset(df_database, root='/')

    print(f"Checking similarity: {len(dataset_query)} new image(s) vs. {len(dataset_database)} existing image(s)...")

    try:
        b_val = min(100, len(dataset_database))
        similarity_matrix = parser_wildfusion_system(dataset_query, dataset_database, B=b_val)

        if similarity_matrix is None or similarity_matrix.size == 0:
            print("Similarity matrix computation failed or returned empty.")
            return False

        max_sim_scores_for_queries = []
        # Convert 1D array to 2D array with single row if needed
        if similarity_matrix.ndim == 1:
            similarity_matrix = similarity_matrix.reshape(1, -1)
            new_image_paths = [new_image_paths[0]]  # Adjust paths list to match matrix shape
            
        if similarity_matrix.shape[0] > 0 and similarity_matrix.shape[1] > 0:
            for i in range(similarity_matrix.shape[0]):
                query_scores = similarity_matrix[i, :]
                max_sim_for_this_query = np.max(query_scores)
                print(f"\nScores for new image {os.path.basename(new_image_paths[i])}:")
                print(f"  Max similarity: {max_sim_for_this_query:.4f}")
                # print("  Individual scores vs existing images:")
                # for j, score in enumerate(query_scores):
                #     print(f"    vs {os.path.basename(existing_horse_image_paths[j])}: {score:.4f}")
                max_sim_scores_for_queries.append(max_sim_for_this_query)
        else:
            print(f"Unexpected similarity matrix shape or size: {similarity_matrix.shape if hasattr(similarity_matrix, 'shape') else 'N/A'}")
            return False
            
        if not max_sim_scores_for_queries:
            print("No similarity scores could be aggregated for queries.")
            return False
        final_average_similarity = np.mean(max_sim_scores_for_queries)

        print(f"\nFinal average max similarity across all images: {final_average_similarity:.4f} (Threshold: {threshold})")
        return final_average_similarity >= threshold
    except Exception as e:
        print(f"Error during similarity check: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_attachments(service, message_id, horse_name_from_subject):
    """Save all images from an email, checking for similarity if the base horse name already exists."""
    downloaded_temp_paths = [] # Keep track of files actually downloaded to temp
    try:
        existing_manifest_df = read_existing_manifest()
        if existing_manifest_df is not None:
            if message_id in existing_manifest_df['message_id'].values:
                print(f"Skipping already processed message: {message_id}")
                return

        os.makedirs(TEMP_DIR, exist_ok=True)
        
        message_gmail_api = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        email_date = extract_oldest_date(message_gmail_api)
        base_name_from_email = horse_name_from_subject
        
        def download_attachments_recursive(parts, msg_id):
            for part in parts:
                if 'parts' in part:
                    download_attachments_recursive(part['parts'], msg_id)
                    continue
                
                if part.get('filename'):
                    if any(part['filename'].lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        original_filename = os.path.basename(part['filename'])
                        safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in original_filename)
                        temp_path = os.path.join(TEMP_DIR, f"temp_{msg_id}_{safe_original_filename}")
                        try:
                            att_id = part['body']['attachmentId']
                            att = service.users().messages().attachments().get(
                                userId='me', messageId=msg_id, id=att_id).execute()
                            file_data = base64.urlsafe_b64decode(att['data'].encode('UTF-8'))
                            with open(temp_path, 'wb') as f: f.write(file_data)
                            downloaded_temp_paths.append((temp_path, original_filename))  # Store tuple of (path, original_name)
                        except Exception as e:
                            print(f"Error downloading attachment {original_filename} from message {msg_id}: {e}")
        
        payload = message_gmail_api.get('payload', {})
        if 'parts' in payload:
            download_attachments_recursive(payload['parts'], message_id)
        elif payload.get('filename') and payload.get('body', {}).get('attachmentId'):
             download_attachments_recursive([payload], message_id)

        if not downloaded_temp_paths:
            print(f"No image attachments found or downloaded for message {message_id}.")
            return

        existing_horses_dict = get_existing_horses_from_manifest(existing_manifest_df)
        final_horse_name_for_saving = base_name_from_email

        if base_name_from_email in existing_horses_dict:
            print(f"Base name '{base_name_from_email}' exists. Checking for duplicate images and similarity...")
            match_found_with_existing_suffix = False
            
            # Filter out any duplicate images based on date and filename
            filtered_temp_paths = []
            for temp_path, original_filename in downloaded_temp_paths:
                is_duplicate = False
                # Extract just the image number/name part (e.g., "IMG_5103" from "IMG_5103.jpg")
                image_base_name = os.path.splitext(original_filename)[0]
                
                # Check against all existing images for this horse
                for suffix, existing_paths in existing_horses_dict[base_name_from_email].items():
                    for existing_path in existing_paths:
                        existing_filename = os.path.basename(existing_path)
                        # Parse the date from the existing filename (format: horsename-YYYYMMDD-messageid-originalname)
                        parts = existing_filename.split('-')
                        if len(parts) >= 2:
                            existing_date = parts[1]
                            existing_image_name = os.path.splitext(parts[-1])[0]  # Get the original image name without extension
                            
                            if existing_date == email_date and existing_image_name == image_base_name:
                                print(f"Skipping duplicate image: {original_filename} (already exists with date {email_date})")
                                is_duplicate = True
                                break
                    if is_duplicate:
                        break
                        
                if not is_duplicate:
                    filtered_temp_paths.append(temp_path)
            
            # Update downloaded_temp_paths to only include non-duplicate images
            if not filtered_temp_paths:
                print("All images are duplicates. Skipping similarity check.")
                return
            
            # Continue with similarity check using filtered paths
            sorted_suffixes = sorted(existing_horses_dict[base_name_from_email].keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

            for suffix in sorted_suffixes:
                current_full_name = f"{base_name_from_email}_{suffix}" if suffix != '1' else base_name_from_email
                existing_image_paths_for_suffix = existing_horses_dict[base_name_from_email][suffix]
                print(f"  Comparing {len(filtered_temp_paths)} new images with {len(existing_image_paths_for_suffix)} images of '{current_full_name}'...")
                if check_horse_similarity(filtered_temp_paths, existing_image_paths_for_suffix):
                    final_horse_name_for_saving = current_full_name
                    print(f"  Similarity match! New images belong to existing individual: {final_horse_name_for_saving}")
                    match_found_with_existing_suffix = True
                    break
            
            if not match_found_with_existing_suffix:
                # Determine the next available suffix number
                max_suffix = 0
                for s_key in existing_horses_dict[base_name_from_email].keys():
                    if s_key.isdigit():
                        max_suffix = max(max_suffix, int(s_key))
                new_suffix_number = max_suffix + 1
                final_horse_name_for_saving = f"{base_name_from_email}_{new_suffix_number}"
                print(f"  No similarity match to existing '{base_name_from_email}' individuals. Assigning new name: {final_horse_name_for_saving}")
        else:
            print(f"New base name '{base_name_from_email}'. Will be saved as {final_horse_name_for_saving} (implicitly suffix _1 if others with this base name are added later).")

        saved_files_info = []
        for temp_path, original_filename in downloaded_temp_paths:  # Unpack the tuple properly
            original_filename_from_email = os.path.basename(original_filename)  # Use original_filename instead of temp_path
            filename_for_manifest = f"{final_horse_name_for_saving}-{email_date}-{message_id}-{original_filename_from_email}"
            final_disk_path = os.path.join(DATASET_DIR, filename_for_manifest)
            
            if os.path.exists(final_disk_path) and not REPLACE_EXISTING:
                print(f"Skipping existing file: {final_disk_path}")
                continue
                
            try:
                os.makedirs(os.path.dirname(final_disk_path), exist_ok=True)
                os.rename(temp_path, final_disk_path)  # temp_path is the actual path to the temp file
                print(f"Saved: {final_disk_path} (Manifest name: {filename_for_manifest})")
                saved_files_info.append({'manifest_filename': filename_for_manifest, 'original_email_filename': original_filename_from_email})
            except OSError as e:
                print(f"Error moving/renaming file {temp_path} to {final_disk_path}: {e}")

        if saved_files_info:
            filenames_for_manifest_update = [info['manifest_filename'] for info in saved_files_info]
            update_manifest_for_email(final_horse_name_for_saving, email_date, message_id, filenames_for_manifest_update, existing_manifest_df)
        else:
            print(f"No new files were ultimately saved for message {message_id}.")

    except Exception as e:
        print(f"Unhandled error processing message {message_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Universal cleanup of any remaining temp files for this message_id
       for temp_path, _ in downloaded_temp_paths:  # Unpack tuple, only need the temp_path
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as oe:
                    print(f"Error during final cleanup of temp file {temp_path}: {oe}")


def read_existing_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            # Specify dtype to avoid mixed type warnings if columns are sometimes empty
            return pd.read_csv(MANIFEST_FILE, dtype={'horse_name': str, 'email_date': str, 'message_id': str, 'filename': str, 'status': str})
        except pd.errors.EmptyDataError:
            print(f"Manifest file '{MANIFEST_FILE}' is empty. A new one will be created if images are processed.")
            return pd.DataFrame(columns=['horse_name', 'email_date', 'message_id', 'filename', 'status'])
        except Exception as e:
            print(f"Error reading manifest file {MANIFEST_FILE}: {e}")
            return None # Indicates an issue reading an existing manifest
    return pd.DataFrame(columns=['horse_name', 'email_date', 'message_id', 'filename', 'status']) # Return empty df if no file


def create_manifest():
    """Create or update manifest by scanning photos directory and preserving existing entries."""
    current_manifest_df = read_existing_manifest()
    if current_manifest_df is None: # Indicates a read error, not just non-existent
        print("Could not read existing manifest properly. Aborting manifest creation/update.")
        return

    # Convert to dictionary for easier management if it's not None
    # key: filename, value: dict of row data
    existing_entries = {row['filename']: row.to_dict() for _, row in current_manifest_df.iterrows()}
    
    processed_filenames_in_dir = set()

    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory {DATASET_DIR} does not exist. No files to scan.")
    else:
        for filename_in_dir in sorted(os.listdir(DATASET_DIR)):
            if any(filename_in_dir.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                processed_filenames_in_dir.add(filename_in_dir)
                if filename_in_dir not in existing_entries: # New file not in manifest
                    try:
                        parts = filename_in_dir.split('-', 3)
                        if len(parts) == 4:
                            horse_name, email_date, message_id, _ = parts[0], parts[1], parts[2], parts[3]
                            new_entry = {
                                'horse_name': horse_name, 'email_date': email_date,
                                'message_id': message_id, 'filename': filename_in_dir, 'status': ''
                            }
                            existing_entries[filename_in_dir] = new_entry
                            print(f"Added new file to manifest: {filename_in_dir}")
                        else:
                            print(f"Could not parse metadata from new filename: {filename_in_dir}. Adding with minimal info.")
                            existing_entries[filename_in_dir] = {
                                'horse_name': 'Unknown', 'email_date': 'Unknown',
                                'message_id': 'Unknown', 'filename': filename_in_dir, 'status': ''
                            }
                    except Exception as e:
                        print(f"Error parsing new filename {filename_in_dir}: {e}")
                # If file is in existing_entries, its data is already there, no need to update unless new logic is added

    # Remove entries from manifest if their file no longer exists in DATASET_DIR
    manifest_filenames_to_remove = [fname for fname in existing_entries if fname not in processed_filenames_in_dir]
    if manifest_filenames_to_remove:
        print(f"Removing {len(manifest_filenames_to_remove)} entries from manifest for files no longer in directory...")
        for fname in manifest_filenames_to_remove:
            del existing_entries[fname]
            print(f"  Removed from manifest: {fname}")

    # Final list of rows for the manifest
    final_manifest_rows = sorted(list(existing_entries.values()), key=lambda x: x['filename'])
    
    try:
        with open(MANIFEST_FILE, 'w', newline='') as f:
            fieldnames = ['horse_name', 'email_date', 'message_id', 'filename', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Ensure all rows have all fieldnames, providing defaults if missing
            for row_dict in final_manifest_rows:
                writer.writerow({field: row_dict.get(field, "") for field in fieldnames})
        print(f"Manifest file updated: {MANIFEST_FILE}")
    except Exception as e:
        print(f"Error writing updated manifest file: {e}")


def update_manifest_for_email(horse_name_to_save, email_date, message_id, manifest_filenames_saved, existing_manifest_df):
    """Update manifest with new files from a single email, merging with existing content."""
    
    # Convert existing manifest to a list of dicts, keyed by filename for easy update/check
    if existing_manifest_df is not None and not existing_manifest_df.empty:
        manifest_dict = {row['filename']: row.to_dict() for _, row in existing_manifest_df.iterrows()}
    else:
        manifest_dict = {}

    for manifest_filename in manifest_filenames_saved:
        # If file already in manifest (e.g. from a previous partial run), update its info, else add new
        entry = manifest_dict.get(manifest_filename, {}) # Get existing or new dict
        entry.update({
            'horse_name': horse_name_to_save,
            'email_date': email_date,
            'message_id': message_id,
            'filename': manifest_filename,
            'status': entry.get('status', '') # Preserve existing status if any, else empty
        })
        manifest_dict[manifest_filename] = entry
    
    # Convert back to list of rows and sort
    updated_manifest_rows = sorted(list(manifest_dict.values()), key=lambda x: x['filename'])
    
    try:
        with open(MANIFEST_FILE, 'w', newline='') as f:
            fieldnames = ['horse_name', 'email_date', 'message_id', 'filename', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row_dict in updated_manifest_rows:
                 writer.writerow({field: row_dict.get(field, "") for field in fieldnames})
        print(f"Manifest updated for email {message_id}, horse {horse_name_to_save}.")
    except Exception as e:
        print(f"Error writing updated manifest after email processing: {e}")


def get_existing_horses_from_manifest(manifest_df):
    """Get dictionary of existing horses and their image paths from manifest"""
    horses = {}  # {base_name: {suffix: [image_paths]}}
    
    if manifest_df is None or manifest_df.empty:
        return horses
        
    for _, row in manifest_df.iterrows():
        # Skip excluded images
        status_val = row.get('status', '') # Handle missing 'status' column gracefully
        if isinstance(status_val, str) and status_val.upper() == 'EXCLUDE':
            continue
            
        horse_name = str(row['horse_name']) # Ensure horse_name is a string
        filename = str(row['filename'])     # Ensure filename is a string

        parts = horse_name.split('_')
        base_name = parts[0]
        suffix = parts[1] if len(parts) > 1 and parts[1].isdigit() else '1' # Default to '1' if no numeric suffix
        
        if base_name not in horses:
            horses[base_name] = {}
        if suffix not in horses[base_name]:
            horses[base_name][suffix] = []
            
        # Construct absolute path for similarity check
        full_path = os.path.abspath(os.path.join(DATASET_DIR, filename))
        if os.path.exists(full_path):
            horses[base_name][suffix].append(full_path)
        else:
            print(f"Warning: Manifest file '{filename}' for horse '{horse_name}' not found at '{full_path}'. Skipping for similarity check.")

    return horses

def main():
    """Main function to process horse emails and save images"""
    print(f"Using DATA_ROOT: {DATA_ROOT}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Manifest file: {MANIFEST_FILE}")
    print(f"Calibration directory: {CALIBRATION_DIR}")
    print(f"Temp directory: {TEMP_DIR}")

    try:
        load_production_system_for_parser()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize the WildFusion system: {e}. Cannot proceed.")
        import traceback
        traceback.print_exc()
        return

    service = gmail_authenticate()
    if not service:
        print("Gmail authentication failed. Cannot proceed.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    manifest_parent_dir = os.path.dirname(MANIFEST_FILE)
    if manifest_parent_dir and not os.path.exists(manifest_parent_dir):
        os.makedirs(manifest_parent_dir, exist_ok=True)


    messages_list = get_horse_emails(service) # Renamed from messages to messages_list
    if not messages_list:
        print("No matching emails found.")
    else:
        print(f"Found {len(messages_list)} emails to process")
        for message_stub in messages_list: # message_stub is a dict from the list e.g. {'id': '...', 'threadId': '...'}
            msg_id = message_stub['id']
            try:
                msg_metadata = service.users().messages().get(userId='me', id=msg_id,
                                                   format='metadata',
                                                   metadataHeaders=['Subject']).execute()
                subject_header = None
                if msg_metadata and 'payload' in msg_metadata and 'headers' in msg_metadata['payload']:
                    subject_header = next((header['value'] for header in msg_metadata['payload']['headers']
                                     if header['name'].lower() == 'subject'), None)
                
                if subject_header:
                    horse_name_from_subject = extract_horse_name(subject_header)
                    if horse_name_from_subject:
                        print(f"\nProcessing email ID: {msg_id}, Subject: {subject_header}, Extracted Base Name: {horse_name_from_subject}")
                        save_attachments(service, msg_id, horse_name_from_subject)
                    else:
                        print(f"Skipping email ID: {msg_id}, Subject: '{subject_header}' (could not extract valid horse name).")
                else:
                    print(f"Skipping email ID: {msg_id} - No subject found in metadata.")
            except HttpError as http_err:
                print(f"HTTP error processing message ID {msg_id}: {http_err}")
                if http_err.resp.status == 404:
                    print(f"  Message ID {msg_id} not found. It might have been deleted.")
                else:
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"General error processing message ID {msg_id}: {e}")
                import traceback
                traceback.print_exc()

    print("\nFinished processing emails. Updating manifest for all files in dataset directory...")
    create_manifest()
    print("\nManifest creation/update complete.")

    print("Attempting to shutdown loky's reusable executor...")
    try:
        from joblib.externals.loky import get_reusable_executor
        executor = get_reusable_executor(kill_workers=True, timeout=30) # timeout for shutdown
        if executor:
            executor.shutdown(wait=True) # wait for shutdown to complete
            print("Loky reusable executor shutdown explicitly.")
        else:
            print("No active loky reusable executor found by joblib's utility.")
    except Exception as e:
        print(f"Error during explicit loky shutdown: {e}")

    print("Checking for active threads before exiting...")
    import threading
    active_thread_count = threading.active_count()
    print(f"Number of active threads: {active_thread_count}")
    for i, thread in enumerate(threading.enumerate()):
        print(f"  Thread {i+1}: Name='{thread.name}', Is Daemon={thread.isDaemon()}, Is Alive={thread.is_alive()}")
        # If you want more details, and know how to get it (e.g. for specific library threads)
        # you might add more introspection here if possible.

    if active_thread_count > 1: # MainThread is always there
        print("Warning: There are non-daemon threads still active which might prevent exit.")
    else:
        print("Only the MainThread (or only daemon threads) appear to be active.")
    
    print("Script attempting to complete.")
    # import sys
    # sys.exit(0) # You could try this as a more standard exit than os._exit
    os._exit(0)

if __name__ == '__main__':
    main()
