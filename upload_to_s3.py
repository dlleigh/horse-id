import os
import sys
import yaml
import boto3
import argparse
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
        
        s3_bucket_name = config['s3']['bucket_name']

        if not os.path.isfile(merged_manifest_file):
            print(f"Error: Merged manifest file not found at '{merged_manifest_file}'")
            sys.exit(1)
        if not os.path.isdir(features_dir):
            print(f"Error: Features directory not found at '{features_dir}'")
            sys.exit(1)
        
        return merged_manifest_file, features_dir, s3_bucket_name
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

def main():
    """Main function to handle argument parsing and trigger the upload."""

    print("Loading configuration...")
    config = load_config()
    merged_manifest_path, features_dir_path, bucket_name = setup_paths(config)
    
    s3_client = boto3.client('s3')

    # 1. Upload the merged manifest file
    print("\n--- Uploading Merged Manifest ---")
    manifest_s3_key = os.path.basename(merged_manifest_path)
    upload_file(s3_client, bucket_name, merged_manifest_path, manifest_s3_key)

    # 2. Upload the contents of the features directory
    print("\n--- Uploading Features Directory ---")
    for root, _, files in os.walk(features_dir_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            features_s3_key = os.path.join('features',filename).replace("\\", "/")
            upload_file(s3_client, bucket_name, local_path, features_s3_key)

    print("\nUpload process finished.")

if __name__ == "__main__":
    main()