import os
import pandas as pd
import yaml
import shutil
import hashlib
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


def generate_message_id_for_subdir(subdir_path, date_str):
    """Generate deterministic message_id for a subdirectory."""
    # Create a hash of the absolute path for uniqueness
    path_hash = hashlib.md5(os.path.abspath(subdir_path).encode()).hexdigest()[:8]
    subdir_name = os.path.basename(subdir_path)
    safe_name = "".join(c if c.isalnum() else "_" for c in subdir_name)
    return f"local-dir-{path_hash}-{safe_name}-{date_str}"


def get_existing_files_for_message_id(manifest_df, message_id):
    """Get set of original_filename values for a given message_id."""
    if manifest_df.empty:
        return set()
    matching_rows = manifest_df[manifest_df['message_id'] == message_id]
    return set(matching_rows['original_filename'].tolist())


def main():
    """Main function to ingest images from subdirectories (one per horse)."""
    print("--- Local Subdirectory Image Ingestion ---")
    print("This script processes a directory containing subdirectories.")
    print("Each subdirectory name will be used as the horse name.")

    # --- Get User Input ---
    while True:
        parent_dir = input("Enter the full path to the directory containing subdirectories (one per horse): ").strip()
        if os.path.isdir(parent_dir):
            break
        print("Error: The provided path is not a valid directory. Please try again.")

    while True:
        email_date_str = input("Enter the date for these photos (YYYYMMDD): ").strip()
        try:
            datetime.strptime(email_date_str, '%Y%m%d')
            break
        except ValueError:
            print("Error: Invalid date format. Please use YYYYMMDD.")

    # --- Process Subdirectories ---
    os.makedirs(DATASET_DIR, exist_ok=True)
    manifest_df = read_or_create_manifest()

    # Find all subdirectories
    subdirs = [d for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')]
    
    if not subdirs:
        print("\nNo subdirectories found in the specified directory. Exiting.")
        return

    print(f"\nFound {len(subdirs)} subdirectories to process: {', '.join(subdirs)}")
    
    total_new_images = 0
    
    for subdir_name in subdirs:
        subdir_path = os.path.join(parent_dir, subdir_name)
        horse_name = subdir_name
        
        print(f"\n--- Processing horse: {horse_name} ---")
        
        # Generate message_id for this subdirectory
        message_id = generate_message_id_for_subdir(subdir_path, email_date_str)
        
        # Check what files already exist for this message_id
        existing_files = get_existing_files_for_message_id(manifest_df, message_id)
        
        # Get all image files in this subdirectory
        image_files = [f for f in os.listdir(subdir_path) 
                      if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        
        # Filter out files that already exist in manifest
        new_image_files = [f for f in image_files if f not in existing_files]
        
        if not new_image_files:
            if image_files:
                print(f"  All {len(image_files)} images already processed for this subdirectory. Skipping.")
            else:
                print(f"  No supported image files found in subdirectory.")
            continue
            
        print(f"  Found {len(new_image_files)} new images to process (out of {len(image_files)} total)")
        
        # Get canonical ID - use existing one if message_id exists, otherwise get next available
        existing_rows = manifest_df[manifest_df['message_id'] == message_id]
        if not existing_rows.empty:
            canonical_id = existing_rows['canonical_id'].iloc[0]
            print(f"  Using existing canonical_id: {canonical_id}")
        else:
            canonical_id = get_next_canonical_id(manifest_df)
            print(f"  Assigned new canonical_id: {canonical_id}")
        
        new_rows = []
        
        for original_filename in new_image_files:
            source_path = os.path.join(subdir_path, original_filename)
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
                    print(f"    Skipping {original_filename}: HEIC support not available")
                    continue
                print(f"    Converting: {original_filename} -> {new_filename_on_disk}")
                if not convert_heic_to_jpg(source_path, final_disk_path):
                    continue
            else:
                print(f"    Copying: {original_filename} -> {new_filename_on_disk}")
                shutil.copy2(source_path, final_disk_path)

            # --- Add to manifest ---
            new_rows.append({
                'horse_name': horse_name,
                'email_date': email_date_str,
                'message_id': message_id,
                'original_filename': original_filename,
                'filename': new_filename_on_disk,
                'date_added': datetime.now().strftime('%Y-%m-%d'),
                'canonical_id': canonical_id,
                'original_canonical_id': canonical_id,
                'last_merged_timestamp': pd.NA,
                'size_ratio': pd.NA,
                'num_horses_detected': '',
                'status': ''
            })

        if new_rows:
            # Update manifest with new rows for this subdirectory
            new_df = pd.DataFrame(new_rows)
            manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
            total_new_images += len(new_rows)
            print(f"    Successfully processed {len(new_rows)} images")

    if total_new_images > 0:
        # Save updated manifest
        manifest_df.to_csv(MANIFEST_FILE, index=False)
        print(f"\n=== SUMMARY ===")
        print(f"Successfully added {total_new_images} new photo entries to the manifest.")
        print(f"Manifest file updated at: {MANIFEST_FILE}")
    else:
        print(f"\n=== SUMMARY ===")
        print("No new images were processed. Manifest unchanged.")
    
    print("\nScript finished.")


if __name__ == '__main__':
    main()

