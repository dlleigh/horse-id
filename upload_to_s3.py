import os
import sys
import yaml
import boto3
import argparse
import pandas as pd
from botocore.exceptions import ClientError

# --- Configuration ---
CONFIG_FILE = 'config.yml'

def load_config():
    """Loads the YAML configuration file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_paths(config):
    """Resolves and validates paths from the configuration."""
    try:
        data_root = os.path.expanduser(config['paths']['data_root'])
        merged_manifest_file = config['paths']['merged_manifest_file'].format(data_root=data_root)
        features_dir = config['paths']['features_dir'].format(data_root=data_root)
        horse_herds_file = config['paths']['horse_herds_file'].format(data_root=data_root)
        
        s3_bucket_name = config['s3']['bucket_name']

        if not os.path.isfile(merged_manifest_file):
            print(f"Error: Merged manifest file not found at '{merged_manifest_file}'")
            sys.exit(1)
        if not os.path.isdir(features_dir):
            print(f"Error: Features directory not found at '{features_dir}'")
            sys.exit(1)
        # Note: horse_herds_file is optional - we'll check existence during upload
        
        return merged_manifest_file, features_dir, horse_herds_file, s3_bucket_name
    except KeyError as e:
        print(f"Error: Missing path configuration for '{e}' in '{CONFIG_FILE}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up paths from config: {e}")
        sys.exit(1)

def upload_file(s3_client, bucket_name, local_path, s3_key):
    """Uploads a single file to an S3 bucket."""
    try:
        print(f"  Uploading {local_path} to s3://{bucket_name}/{s3_key}...")
        s3_client.upload_file(local_path, bucket_name, s3_key)
    except ClientError as e:
        print(f"    ERROR: Failed to upload {local_path}. Reason: {e}")
        return False
    except FileNotFoundError:
        print(f"    ERROR: Local file not found: {local_path}")
        return False
    return True

def clean_manifest_for_upload(manifest_path):
    """
    Remove the segmentation_mask column from the manifest file before uploading.
    Returns the path to the cleaned manifest file.
    """
    try:
        # Read the manifest file
        df = pd.read_csv(manifest_path)
        
        # Check if segmentation_mask column exists
        if 'segmentation_mask' in df.columns:
            print(f"  Removing 'segmentation_mask' column from manifest...")
            df = df.drop('segmentation_mask', axis=1)
            
            # Create a temporary cleaned manifest file
            cleaned_manifest_path = manifest_path.replace('.csv', '_cleaned.csv')
            df.to_csv(cleaned_manifest_path, index=False)
            print(f"  Created cleaned manifest at: {cleaned_manifest_path}")
            return cleaned_manifest_path
        else:
            print(f"  No 'segmentation_mask' column found in manifest")
            return manifest_path
            
    except Exception as e:
        print(f"  Warning: Could not clean manifest file: {e}")
        print(f"  Uploading original manifest file")
        return manifest_path

def main():
    """Main function to handle argument parsing and trigger the upload."""

    print("Loading configuration...")
    config = load_config()
    merged_manifest_path, features_dir_path, horse_herds_path, bucket_name = setup_paths(config)
    
    s3_client = boto3.client('s3')

    # 1. Clean and upload the merged manifest file
    print("\n--- Uploading Merged Manifest ---")
    cleaned_manifest_path = clean_manifest_for_upload(merged_manifest_path)
    manifest_s3_key = os.path.basename(merged_manifest_path)  # Use original name for S3 key
    upload_success = upload_file(s3_client, bucket_name, cleaned_manifest_path, manifest_s3_key)
    
    # Clean up temporary file if it was created
    if cleaned_manifest_path != merged_manifest_path:
        try:
            os.remove(cleaned_manifest_path)
            print(f"  Cleaned up temporary file: {cleaned_manifest_path}")
        except Exception as e:
            print(f"  Warning: Could not remove temporary file {cleaned_manifest_path}: {e}")

    # 2. Upload horse herds file if it exists
    print("\n--- Uploading Horse Herds File ---")
    if os.path.exists(horse_herds_path):
        horse_herds_s3_key = os.path.basename(horse_herds_path)
        upload_success = upload_file(s3_client, bucket_name, horse_herds_path, horse_herds_s3_key)
        if upload_success:
            print(f"  Successfully uploaded horse herds file")
        else:
            print(f"  Failed to upload horse herds file")
    else:
        print(f"  Horse herds file not found at: {horse_herds_path}")
        print(f"  Skipping horse herds upload. Run parse_horse_herds.py first to generate this file.")

    # 3. Upload the contents of the features directory
    print("\n--- Uploading Features Directory ---")
    for root, _, files in os.walk(features_dir_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            features_s3_key = os.path.join('features',filename).replace("\\", "/")
            upload_file(s3_client, bucket_name, local_path, features_s3_key)

    print("\nUpload process finished.")

if __name__ == "__main__":
    main()