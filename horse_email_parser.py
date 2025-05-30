import os.path
import pickle # For older versions of the library, token.pickle was common
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import base64
import re
from datetime import datetime
from email.message import EmailMessage
from dateutil import parser, tz

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly'] # Read-only access is sufficient for testing labels

TOKEN_FILE = 'token.json' # Stores the user's access and refresh tokens
CREDENTIALS_FILE = 'credentials.json' # Your OAuth 2.0 credentials

DATASET_DIR = 'datasets/horse_photo_extractor' # Main directory for storing horse photos

def gmail_authenticate():
    """Shows basic usage of the Gmail API.
    Performs authentication and returns the Gmail API service object.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"Error loading token from {TOKEN_FILE}: {e}")
            creds = None # Ensure creds is None if loading fails

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("Credentials expired. Refreshing token...")
                creds.refresh(Request())
                print("Token refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing token: {e}")
                creds = None # Force re-authentication
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"ERROR: Credentials file '{CREDENTIALS_FILE}' not found.")
                print("Please download it from Google Cloud Console and place it in the same directory as this script.")
                return None
            try:
                print("No valid token found. Starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                # Open a browser for the user to authenticate
                # The port number here is arbitrary; you can change it if 8080 is in use.
                creds = flow.run_local_server(port=8080) # This will open a browser window
                print("OAuth flow completed. Credentials obtained.")
            except Exception as e:
                print(f"Error during OAuth flow: {e}")
                return None
        # Save the credentials for the next run
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


def get_horse_emails(service, query="subject:fall OR subject:spring OR subject:summer OR subject:winter"):
    """Fetch emails that likely contain horse photos based on seasonal keywords"""
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        return messages
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def create_horse_folder(horse_name):
    """Create a folder for the horse if it doesn't exist"""
    folder_path = os.path.join(DATASET_DIR, horse_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def extract_horse_name(subject):
    """Extract horse name from email subject"""
    # Remove 'Fwd: ' prefix if present
    subject = re.sub(r'^Fwd?:\s*', '', subject, flags=re.IGNORECASE)
    
    # Match pattern like "Oliver- fall 24'" or "Oliver - Spring 2024"
    match = re.match(r'^([^-]+)-?\s*(?:fall|spring|summer|winter).*$', subject, re.IGNORECASE)
    if match:
        horse_name = match.group(1)
        horse_name = re.sub(r'[^\w\s-]', '', horse_name).strip()  # Clean the horse name
        return horse_name
    return None

def extract_oldest_date(message):
    """Extract the oldest sent date from email headers and forwarded content"""
    dates = []
    
    tzinfos = {
        "CDT": tz.gettz("America/Chicago"),
        "CST": tz.gettz("America/Chicago"),
        "EDT": tz.gettz("America/New_York"),
        "EST": tz.gettz("America/New_York"),
        "PDT": tz.gettz("America/Los_Angeles"),
        "PST": tz.gettz("America/Los_Angeles"),
        "MDT": tz.gettz("America/Denver"),
        "MST": tz.gettz("America/Denver")
    }

    def process_part(part):
        """Helper function to recursively process message parts"""
        # If this part has nested parts, process them
        if 'parts' in part:
            for subpart in part['parts']:
                process_part(subpart)
            return

        # Look for text content
        if part.get('mimeType') == 'text/plain':
            if 'data' in part.get('body', {}):
                body_bytes = base64.urlsafe_b64decode(part['body']['data'])
                body_text = body_bytes.decode('utf-8')
                
                # Look for lines that might contain dates
                header_patterns = [
                    r'(?:Sent|Date|From):.+$',
                    r'On.+wrote:$',
                    r'>.+\d{4}.*$'
                ]
                
                for line in body_text.split('\n'):
                    line = line.strip()
                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in header_patterns):
                        try:
                            # Parse date and keep only the date portion
                            parsed_date = parser.parse(line, fuzzy=True, tzinfos=tzinfos)
                            # Convert to date object (strips time and timezone)
                            date_only = parsed_date.date()
                            if date_only.year > 1970:  # Sanity check
                                dates.append(date_only)
                                # print(f"Found date: {date_only} in line: {line}")  # Debug line
                        except (ValueError, parser.ParserError):
                            continue

    # Start processing from the message payload
    payload = message.get('payload', {})
    process_part(payload)
    
    # Add the email's internal date as fallback (date portion only)
    internal_date = datetime.fromtimestamp(int(message.get('internalDate', 0)) / 1000).date()
    dates.append(internal_date)
    
    # Return the oldest date found
    oldest_date = min(dates) if dates else internal_date
    return oldest_date.strftime('%Y%m%d')  # Format changed to date only

def save_attachments(service, message_id, horse_folder, horse_name):
    """Save image attachments from an email to the horse's folder"""
    try:
        # Get the full message
        message = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        
        # Get the oldest date from the email chain
        email_date = extract_oldest_date(message)
        
        def process_parts(parts):
            """Helper function to recursively process message parts"""
            for part in parts:
                # If this part has nested parts, process them
                if 'parts' in part:
                    process_parts(part['parts'])
                    continue
                
                # Check if this part has a filename and is an image
                if part.get('filename'):
                    if any(part['filename'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                        if 'body' in part and 'attachmentId' in part['body']:
                            att_id = part['body']['attachmentId']
                            
                            # Get the attachment
                            att = service.users().messages().attachments().get(
                                userId='me', messageId=message_id, id=att_id).execute()
                            
                            # Decode the attachment data
                            file_data = base64.urlsafe_b64decode(att['data'].encode('UTF-8'))
                            
                            # Create filename with email timestamp instead of current time
                            filename = part['filename']
                            new_filename = f"{horse_name}-{email_date}-{filename}"
                            filepath = os.path.join(horse_folder, new_filename)
                            
                            # Save the file
                            with open(filepath, 'wb') as f:
                                f.write(file_data)
                            print(f"Saved: {filepath}")
        
        # Get message payload and start processing
        payload = message.get('payload', {})
        if 'parts' in payload:
            process_parts(payload['parts'])
        else:
            process_parts([payload])
                        
    except Exception as e:
        print(f"Error processing message {message_id}: {e}")

def main():
    """
    Authenticates with Gmail and lists the user's labels.
    """
    print("Attempting to authenticate with Gmail...")
    service = gmail_authenticate()

    if service:
        print("\nSuccessfully authenticated and Gmail service created.")
    else:
        print("\nOAuth 2.0 setup test failed: Could not authenticate or create Gmail service.")

    """Main function to process horse emails and save images"""
    # Create main dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Get relevant emails
    messages = get_horse_emails(service)
    if not messages:
        print("No matching emails found")
        return

    print(f"Found {len(messages)} emails to process")
    
    for message in messages:
        try:
            # Get full message details
            msg = service.users().messages().get(userId='me', id=message['id'], format='metadata',
                                               metadataHeaders=['Subject']).execute()
            subject = next((header['value'] for header in msg['payload']['headers'] 
                          if header['name'] == 'Subject'), None)
            
            if subject:
                horse_name = extract_horse_name(subject)
                if horse_name:
                    if horse_name != "Moe":
                        continue
                    horse_folder = create_horse_folder(horse_name)
                    print(f"\nProcessing email: {subject}")
                    save_attachments(service, message['id'], horse_folder, horse_name)
                else:
                    print(f"Skipping email with invalid subject format: {subject}")
        except Exception as e:
            print(f"Error processing message: {e}")

if __name__ == '__main__':
    main()