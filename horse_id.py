import argparse
import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import tempfile
import pickle
import yaml
import sys
import boto3 
from botocore.exceptions import ClientError
import json 

from twilio.rest import Client
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import urllib.parse 

from wildlife_datasets import datasets
from wildlife_tools.inference import TopkClassifier
import torchvision.transforms as T
import timm
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset
from wildlife_tools.similarity import CosineSimilarity

# --- Configuration ---
CONFIG_FILE = 'config.yml'

class Horses(datasets.WildlifeDataset):
    def __init__(self, root_dir, manifest_file_path):
        self.manifest_file_path = manifest_file_path
        super().__init__(root_dir, check_files=False)

    def create_catalogue(self) -> pd.DataFrame:
        """Create catalogue from manifest file"""
        manifest_df = pd.read_csv(self.manifest_file_path)
        
        rows = []
        for _, row in manifest_df.iterrows():
            if 'status' in row and row['status'] == 'EXCLUDE':
                continue
            if 'num_horses_detected' in row and row['num_horses_detected'] in ['NONE', 'MULTIPLE']:
                continue
            rows.append({
                'image_id': row['filename'],
                #'identity': row['canonical_id'],
                'identity': row['horse_name'],
                'path': row['filename'],
                'date': pd.to_datetime(str(row['email_date']), format='%Y%m%d')
            })
        
        df = pd.DataFrame(rows)
        result = self.finalize_catalogue(df)
        return result

def load_config():
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        # In Lambda, we don't sys.exit directly from a helper, but let the handler return an error.
        raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_paths(config):
    try:
        # Use data_root from config, but allow override via an environment variable for Docker.
        # This makes the script more portable.
        data_root_config = os.environ.get('HORSE_ID_DATA_ROOT', config['paths']['data_root'])
        data_root = os.path.expanduser(data_root_config)
        manifest_file = config['paths']['merged_manifest_file'].format(data_root=data_root)
        features_dir = config['paths']['features_dir'].format(data_root=data_root)
        s3_bucket_name = config['s3']['bucket_name']
        return manifest_file, features_dir, s3_bucket_name
    except KeyError as e:
        logger.error(f"Error: Missing path configuration for '{e}' in '{CONFIG_FILE}'.")
        raise ValueError(f"Missing path configuration: {e}")
    except Exception as e:
        logger.error(f"Error setting up paths from config: {e}")
        raise RuntimeError(f"Error setting up paths: {e}")

def download_from_s3(s3_client, bucket_name, s3_key, local_path):
    """Downloads a file from S3 if it doesn't exist locally."""
    if os.path.exists(local_path):
        logger.info(f"  File already exists locally: {local_path}")
        return True

    logger.info(f"  Downloading s3://{bucket_name}/{s3_key} to {local_path}...")
    try:
        # Ensure local directory exists
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info("  Download complete.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error(f"    ERROR: The file was not found in S3: s3://{bucket_name}/{s3_key}")
        else:
            logger.error(f"    ERROR: Failed to download file from S3. Reason: {e}")
        return False

def process_image_for_identification(image_url):
    """
    Core logic to download image, extract features, and identify horse.
    Returns a dictionary of prediction results.
    """
    config = load_config()
    manifest_file, features_dir, s3_bucket_name = setup_paths(config)

    # --- S3 Download Logic ---
    logger.info("Checking for required files from S3...")
    s3_client = boto3.client('s3')

    # 1. Download manifest file
    manifest_s3_key = os.path.basename(manifest_file)
    if not download_from_s3(s3_client, s3_bucket_name, manifest_s3_key, manifest_file):
        raise RuntimeError("Could not retrieve manifest file.")

    # 2. Download features file
    features_local_path = os.path.join(features_dir, 'database_deep_features.pkl')
    # Construct S3 key based on upload script logic: features_dir_basename/filename
    features_s3_key = os.path.join(os.path.basename(features_dir), 'database_deep_features.pkl').replace("\\", "/")
    if not download_from_s3(s3_client, s3_bucket_name, features_s3_key, features_local_path):
        raise RuntimeError("Could not retrieve features file.")
    logger.info("--------------------------------------------")

    if not os.path.isfile(manifest_file):
        raise FileNotFoundError(f"MANIFEST_FILE '{manifest_file}' not found.")
    if not os.path.isdir(features_dir):
        raise NotADirectoryError(f"FEATURES_DIR '{features_dir}' not found or not a directory.")

    logger.info("Initializing Horse dataset...")
    horses_dataset_obj = Horses(None, manifest_file_path=manifest_file)
    horses_df_all = horses_dataset_obj.create_catalogue()
    if horses_df_all.empty:
        raise ValueError("The horse catalogue is empty. Check manifest file and Horses class.")

    dataset_database = ImageDataset(horses_df_all, horses_dataset_obj.root)

    logger.info(f"Downloading image from {image_url}...")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img_bytes = io.BytesIO(response.content)
        # Validate if the downloaded content can be opened as an image by PIL
        try:
            Image.open(img_bytes).verify() # verify() checks for corruption
            img_bytes.seek(0) # Reset stream position after verify
        except (IOError, SyntaxError) as e:
            raise ValueError(f"Downloaded content from {image_url} is not a valid image or is corrupted: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading image: {e}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(img_bytes.getvalue())
        temp_image_path = tmp_file.name
    
    logger.info(f"Temporary image saved to: {temp_image_path}")

    try:
        query_df = pd.DataFrame([{
            'path': os.path.basename(temp_image_path), # ImageDataset needs relative path from its root
            'identity': -1, # Dummy identity for query
            'image_id': 'query_image',
            'horse_name': 'query_horse', # Dummy name
            'date': pd.Timestamp.now() # Dummy date
        }])
        transform = T.Compose([T.Resize([384, 384]), T.ToTensor()])
        dataset_query_single = ImageDataset(query_df, os.path.dirname(temp_image_path), transform=transform)

        logger.info("Extracting features for query image...")

        logger.info("Attempting to create timm model backbone...")
        backbone = timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', pretrained=True, cache_dir='/tmp/model_cache')
        logger.info("Timm model backbone created successfully.")
        extractor = DeepFeatures(backbone)
        query_features = extractor(dataset_query_single)

        features_output_path = os.path.join(features_dir, 'database_deep_features.pkl')
        logger.info(f"Loading database features from {features_output_path}...")
        with open(features_output_path, 'rb') as f:
            database_features = pickle.load(f)

        logger.info("Calculating similarity...")
        similarity_function = CosineSimilarity()
        similarity = similarity_function(query_features, database_features)
        
        db_labels = dataset_database.labels_string

        classifier = TopkClassifier(k=5, database_labels=db_labels, return_all=True)
        predictions,scores,idx = classifier(similarity)

        # Load the confidence threshold from the configuration
        CONFIDENCE_THRESHOLD = config['similarity']['inference_threshold']

        results = {
            "status": "success",
            "query_image_url": image_url,
            "predictions": []
        }
        logger.info("\n--- Predictions above Confidence Threshold ---")        
        found_above_threshold = False
        for pred, score in zip(predictions[0], scores[0]):
            if score > CONFIDENCE_THRESHOLD:
                logger.info(f"  Predicted identity: {pred}, Score: {score:.4f}")
                results["predictions"].append({"identity": pred, "score": round(score, 4), "above_threshold": True})
                found_above_threshold = True
            else:
                results["predictions"].append({"identity": pred, "score": round(score, 4), "above_threshold": False})
        if not found_above_threshold:
            logger.info(f"  No predictions found above the confidence threshold of {CONFIDENCE_THRESHOLD}.")
            logger.info("  Displaying top 5 predictions regardless of threshold:")
            for pred_data in results["predictions"]:
                logger.info(f"  identity: {pred_data['identity']}, Score: {pred_data['score']:.4f}")
        logger.info("--------------------------------------------")
        
        return results

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logger.info(f"Cleaned up temporary image: {temp_image_path}")

def _parse_twilio_event(event):
    """Helper function to parse the incoming event from API Gateway."""
    logger.info(f"Received raw event from Lambda: {json.dumps(event)}")
    incoming_payload = {}
    if 'body' in event and event['body']:
        try:
            content_type = event.get('headers', {}).get('Content-Type', '').lower()
            if 'application/x-www-form-urlencoded' in content_type:
                parsed_qs = urllib.parse.parse_qs(event['body'])
                incoming_payload = {k: v[0] for k, v in parsed_qs.items()}
                logger.info(f"Parsed event body (form-urlencoded): {json.dumps(incoming_payload)}")
            elif 'application/json' in content_type:
                incoming_payload = json.loads(event['body'])
                logger.info(f"Parsed event body (JSON): {json.dumps(incoming_payload)}")
            else:
                logger.warning(f"Unexpected Content-Type: {content_type}. Attempting to use raw event body as payload.")
                incoming_payload = event['body']
        except Exception as e:
            logger.error(f"Error parsing event body: {e}. Falling back to raw event as payload.")
            incoming_payload = event
    else:
        incoming_payload = event
        logger.info("No 'body' field found in event. Assuming direct payload invocation.")
    return incoming_payload

def twilio_webhook_handler(event, context):
    """
    Receives the initial webhook from Twilio, invokes the processor asynchronously,
    and returns an immediate response to Twilio.
    This should be the handler for your 'twilio-webhook-responder' Lambda.
    """
    logger.info("--- twilio_webhook_handler invoked ---")
    
    processor_lambda_name = os.environ.get('PROCESSOR_LAMBDA_NAME')
    if not processor_lambda_name:
        logger.error("FATAL: PROCESSOR_LAMBDA_NAME environment variable not set.")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/xml'},
            'body': '<Response><Message>Error: System is not configured correctly.</Message></Response>'
        }

    try:
        lambda_client = boto3.client('lambda')
        logger.info(f"Asynchronously invoking processor: {processor_lambda_name}")
        lambda_client.invoke(
            FunctionName=processor_lambda_name,
            InvocationType='Event',
            Payload=json.dumps(event)
        )
        logger.info("Processor invocation successful.")
    except Exception as e:
        logger.exception(f"Failed to invoke processor lambda: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/xml'},
            'body': f'<Response><Message>Error: Could not start identification process.</Message></Response>'
        }

    response_message = (
        "Identification started! (Please note: By sending a picture to the Little Tree Farms Horse ID number, "
        "you consent to receive a response. Respond STOP to stop receiving messages.)"
    )
    twilio_response_body = f'<Response><Message>{response_message}</Message></Response>'

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/xml'},
        'body': twilio_response_body
    }

def horse_id_processor_handler(event, context):
    """
    Invoked asynchronously. Performs the actual image identification and sends
    the results back to the user via the Twilio API.
    This should be the handler for your 'horse-id-processor' Lambda.
    """
    logger.info("--- horse_id_processor_handler invoked ---")
    incoming_payload = _parse_twilio_event(event)

    image_url = incoming_payload.get('MediaUrl0')
    if not image_url:
        logger.error("No image URL found in the payload. Aborting.")
        return {'statusCode': 400, 'body': 'No image URL found.'}

    try:
        prediction_results = process_image_for_identification(image_url)
        
        response_message = "Horse Identification Results:\n"
        found_match = False
        for pred in prediction_results["predictions"]:
            if pred["above_threshold"]:
                response_message += f"  Match: {pred['identity']} (Score: {pred['score']})\n"
                found_match = True
        
        if not found_match:
            response_message = "No strong match found. Top predictions:\n"
            for pred in prediction_results["predictions"]:
                response_message += f"  {pred['identity']} (Score: {pred['score']})\n"

        logger.info("Sending final results via Twilio API...")
        config = load_config()
        twilio_config = config.get('twilio', {})
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

        if not account_sid or not auth_token:
            raise ValueError("Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) not found.")

        client = Client(account_sid, auth_token)
        twilio_sending_number = incoming_payload.get('To')
        user_number = incoming_payload.get('From')

        if not twilio_sending_number or not user_number:
            raise ValueError("To/From numbers not found in payload.")

        message = client.messages.create(body=response_message, from_=twilio_sending_number, to=user_number)
        logger.info(f"Successfully sent SMS with SID: {message.sid}")
        return {'statusCode': 200, 'body': json.dumps({'message_sid': message.sid})}
    except Exception as e:
        logger.exception(f"An error occurred during horse identification processing: {e}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify a horse from an image URL (tests the core processing logic).")
    parser.add_argument("image_url", type=str, help="URL of the horse image to identify.")
    
    args = parser.parse_args()
    
    print("\n--- Running Core Identification Logic (simulated) ---")
    try:
        results = process_image_for_identification(args.image_url)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")
        
    sys.exit(0)
