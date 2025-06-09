import os
import base64
import re
import csv
import yaml
from datetime import datetime
from dateutil import parser, tz
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Load Configuration ---
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# --- Use the config values ---
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
DATASET_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
MANIFEST_FILE = config['paths']['manifest_file'].format(data_root=DATA_ROOT)
TEMP_DIR = config['paths']['temp_dir'].format(data_root=DATA_ROOT)
GMAIL_CONFIG = config['gmail']
TOKEN_FILE = GMAIL_CONFIG['token_file']
CREDENTIALS_FILE = GMAIL_CONFIG['credentials_file']
SCOPES = GMAIL_CONFIG['scopes']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']


def gmail_authenticate():
    """Performs authentication and returns the Gmail API service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_emails_to_process(service, existing_message_ids):
    """Fetch all emails that have not yet been processed."""
    try:
        messages = []
        next_page_token = None
        
        while True:
            results = service.users().messages().list(
                userId='me', 
                q='', 
                pageToken=next_page_token,
                maxResults=500  # Get maximum allowed per request
            ).execute()
            
            if 'messages' in results:
                messages.extend(results['messages'])
            
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break
                
        # Filter out messages that are already in our manifest
        new_messages = [m for m in messages if m['id'] not in existing_message_ids]
        return new_messages
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def extract_horse_name(subject):
    """Extract horse name from email subject."""
    subject = re.sub(r'^Fwd?:\s*', '', subject, flags=re.IGNORECASE)
    match = re.match(r'^([^-]+)-?\s*(?:fall|spring|summer|winter).*$', subject, re.IGNORECASE)
    if match:
        horse_name = match.group(1).strip()
        return re.sub(r'[^\w\s-]', '', horse_name).strip()
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
                # Insert newline before each instance of "To", "From", "Sent", and "Subject"
                body_text = re.sub(r'(?m)(To|From|Sent|Subject):', r'\n\1:', body_text)
                header_patterns = [r'(?:Sent|Date|From):.+', r'On.+wrote:$', r'>.+\d{4}.*$']
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

def get_next_canonical_id(df):
    """Finds the maximum existing canonical_id and returns the next integer."""
    if df.empty or 'canonical_id' not in df.columns or df['canonical_id'].isna().all():
        return 1
    return int(df['canonical_id'].max()) + 1

def read_or_create_manifest():
    """Reads the manifest if it exists, otherwise creates an empty DataFrame."""
    if os.path.exists(MANIFEST_FILE):
        try:
            return pd.read_csv(MANIFEST_FILE, dtype={'message_id': str, 'canonical_id': 'Int64'})
        except pd.errors.EmptyDataError:
            pass # Will return empty df below
    return pd.DataFrame(columns=['horse_name', 'email_date', 'message_id', 'filename',
                                 'date_added', 'canonical_id', 'num_horses_detected', 'status'])

def save_attachments(service, message_stub, manifest_df):
    """
    Downloads attachments from a single email, assigns a new canonical_id
    for this group of photos, and returns the new manifest rows.
    """
    msg_id = message_stub['id']
    new_rows = []
    try:
        msg_full = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        subject_header = next((h['value'] for h in msg_full['payload']['headers'] if h['name'].lower() == 'subject'), "")
        horse_name = extract_horse_name(subject_header)
        if not horse_name:
            print(f"Skipping email ID: {msg_id}, Subject: '{subject_header}' (no valid horse name).")
            return []

        email_date = extract_oldest_date(msg_full)
        attachments = []

        def find_attachments(parts):
            for part in parts:
                if part.get('filename') and any(part['filename'].lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    attachments.append(part)
                if 'parts' in part:
                    find_attachments(part['parts'])

        if 'parts' in msg_full['payload']:
            find_attachments(msg_full['payload']['parts'])

        if not attachments:
            return []

        print(f"Processing email ID: {msg_id}, Subject: '{subject_header}', Extracted Name: '{horse_name}'")
        # All photos in one email get the same new canonical_id
        next_id = get_next_canonical_id(manifest_df) + len(new_rows)

        for part in attachments:
            att_id = part['body']['attachmentId']
            att = service.users().messages().attachments().get(userId='me', messageId=msg_id, id=att_id).execute()
            file_data = base64.urlsafe_b64decode(att['data'].encode('UTF-8'))

            original_filename = part['filename']
            filename_for_disk = f"{horse_name}-{email_date}-{msg_id}-{original_filename}"
            final_disk_path = os.path.join(DATASET_DIR, filename_for_disk)

            os.makedirs(DATASET_DIR, exist_ok=True)
            with open(final_disk_path, 'wb') as f:
                f.write(file_data)
            print(f"  Saved: {final_disk_path}")

            new_rows.append({
                'horse_name': horse_name,
                'email_date': email_date,
                'message_id': msg_id,
                'filename': filename_for_disk,
                'date_added': datetime.now().strftime('%Y-%m-%d'),
                'canonical_id': next_id,
                'num_horses_detected': '',
                'status': ''
            })
    except HttpError as e:
        print(f"Error processing message {msg_id}: {e}")
    return new_rows


def main():
    """Main function to process emails and create initial manifest entries."""
    print(f"Using manifest file: {MANIFEST_FILE}")
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)

    manifest_df = read_or_create_manifest()
    existing_ids = set(manifest_df['message_id'].dropna())

    service = gmail_authenticate()
    if not service:
        print("Gmail authentication failed. Cannot proceed.")
        return

    messages_to_process = get_emails_to_process(service, existing_ids)
    if not messages_to_process:
        print("No new emails to process.")
        return

    print(f"Found {len(messages_to_process)} new emails to process.")
    total_new_rows = 0
    
    for i, message_stub in enumerate(messages_to_process, 1):
        new_rows = save_attachments(service, message_stub, manifest_df)
        if new_rows:
            # Update the dataframe with new rows
            manifest_df = pd.concat([manifest_df, pd.DataFrame(new_rows)], ignore_index=True)
            # Save after each email is processed
            manifest_df.to_csv(MANIFEST_FILE, index=False)
            total_new_rows += len(new_rows)
            print(f"Progress: {i}/{len(messages_to_process)} emails processed. Total new photos: {total_new_rows}")

    print(f"\nAll done! Added {total_new_rows} new photo entries to manifest.")
    print("\nScript finished.")

if __name__ == '__main__':
    main()