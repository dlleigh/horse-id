import os
import pandas as pd
import yaml
import shutil
from datetime import datetime
from PIL import Image

# Attempt to import pillow_heif for HEIC support.
try:
    import pillow_heif
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# --- Load Configuration ---
try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yml not found. Please ensure the configuration file exists in the same directory.")
    exit()

# --- Use the config values ---
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
DATASET_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
MANIFEST_FILE = config['paths']['manifest_file'].format(data_root=DATA_ROOT)

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']
if HEIC_SUPPORT:
    IMAGE_EXTENSIONS.extend(['.heic', '.heif'])
else:
    print("\nWarning: `pillow-heif` is not installed. HEIC/HEIF files will be ignored.")
    print("To enable HEIC support, run: pip install pillow-heif\n")


def read_or_create_manifest():
    """Reads the manifest if it exists, otherwise creates an empty DataFrame."""
    if os.path.exists(MANIFEST_FILE):
        try:
            # Ensure message_id is read as a string to avoid type issues
            return pd.read_csv(MANIFEST_FILE, dtype={'message_id': str, 'canonical_id': 'Int64', 'original_canonical_id': 'Int64'})
        except pd.errors.EmptyDataError:
            pass  # Will return empty df below
    return pd.DataFrame(columns=['horse_name', 'email_date', 'message_id', 'original_filename',
                                 'filename', 'date_added', 'canonical_id', 'original_canonical_id', 'size_ratio',
                                 'num_horses_detected', 'last_merged_timestamp', 'status'])


def get_next_canonical_id(df):
    """Finds the maximum existing canonical_id and returns the next integer."""
    if df.empty or 'canonical_id' not in df.columns or df['canonical_id'].isna().all():
        return 1
    return int(df['canonical_id'].max()) + 1


def convert_heic_to_jpg(heic_path, jpg_path):
    """Converts a HEIC image to JPG and saves it."""
    try:
        heif_file = pillow_heif.read_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        image.save(jpg_path, "JPEG")
        return True
    except Exception as e:
        print(f"  - Could not convert {os.path.basename(heic_path)}: {e}")
        return False


def main():
    """Main function to ingest images from a local directory."""
    print("--- Local Directory Image Ingestion ---")

    # --- Get User Input ---
    while True:
        source_dir = input("Enter the full path to the directory with images: ").strip()
        if os.path.isdir(source_dir):
            break
        print("Error: The provided path is not a valid directory. Please try again.")

    horse_name = input("Enter the horse's name: ").strip()

    while True:
        email_date_str = input("Enter the date for these photos (YYYYMMDD): ").strip()
        try:
            datetime.strptime(email_date_str, '%Y%m%d')
            break
        except ValueError:
            print("Error: Invalid date format. Please use YYYYMMDD.")

    # --- Process Images ---
    os.makedirs(DATASET_DIR, exist_ok=True)
    manifest_df = read_or_create_manifest()

    # Generate a unique ID for this ingestion batch
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_horse_name = "".join(c if c.isalnum() else "_" for c in horse_name)
    message_id = f"local-{timestamp}-{safe_horse_name}"

    # Get the next canonical ID for this group of photos
    next_id = get_next_canonical_id(manifest_df)

    new_rows = []
    image_files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print("\nNo supported image files found in the directory. Exiting.")
        return

    print(f"\nFound {len(image_files)} supported images. Processing...")

    for original_filename in image_files:
        source_path = os.path.join(source_dir, original_filename)
        base, ext = os.path.splitext(original_filename)
        ext_lower = ext.lower()

        # Define the new filename for the dataset directory
        if ext_lower in ['.heic', '.heif']:
            new_filename_on_disk = f"{message_id}-{base}.jpg"
        else:
            new_filename_on_disk = f"{message_id}-{original_filename}"

        final_disk_path = os.path.join(DATASET_DIR, new_filename_on_disk)

        # --- Copy or Convert File ---
        if ext_lower in ['.heic', '.heif']:
            if not HEIC_SUPPORT:
                continue # Skip if library not installed
            print(f"  Converting: {original_filename} -> {new_filename_on_disk}")
            if not convert_heic_to_jpg(source_path, final_disk_path):
                continue # Skip if conversion fails
        else:
            print(f"  Copying: {original_filename} -> {new_filename_on_disk}")
            shutil.copy2(source_path, final_disk_path)

        # --- Add to manifest ---
        new_rows.append({
            'horse_name': horse_name,
            'email_date': email_date_str,
            'message_id': message_id,
            'original_filename': original_filename,
            'filename': new_filename_on_disk,
            'date_added': datetime.now().strftime('%Y-%m-%d'),
            'canonical_id': next_id,
            'original_canonical_id': next_id,
            'last_merged_timestamp': pd.NA,
            'size_ratio': pd.NA,
            'num_horses_detected': '',
            'status': ''
        })

    if not new_rows:
        print("\nNo images were successfully processed. Manifest not updated.")
        return

    # --- Update and Save Manifest ---
    new_df = pd.DataFrame(new_rows)
    updated_manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
    updated_manifest_df.to_csv(MANIFEST_FILE, index=False)

    print(f"\nSuccess! Added {len(new_rows)} new photo entries to the manifest.")
    print(f"All photos in this batch were assigned canonical_id: {next_id}")
    print(f"Manifest file updated at: {MANIFEST_FILE}")
    print("\nScript finished.")


if __name__ == '__main__':
    main()

