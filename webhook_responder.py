import json
import os
import boto3
import logging
import urllib.parse
import base64
from twilio.request_validator import RequestValidator

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def webhook_handler(event, context):
    """
    Receives the initial webhook from Twilio, invokes the processor asynchronously,
    and returns an immediate response to Twilio.
    This should be the handler for your 'webhook-responder' Lambda.
    """
    print(json.dumps(event))
    logger.info("--- webhook_handler invoked ---")

    # --- Twilio Request Validation ---
    # This is crucial for security to ensure the request genuinely came from Twilio.
    # Get Twilio Auth Token from environment variables (recommended for secrets)
    twilio_auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    if not twilio_auth_token:
        logger.error("FATAL: TWILIO_AUTH_TOKEN environment variable not set. Cannot validate Twilio request.")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/xml'},
            'body': '<Response><Message>Error: Server configuration issue (missing Twilio Auth Token).</Message></Response>'
        }

    validator = RequestValidator(twilio_auth_token)

    # Extract necessary components from the API Gateway event
    # API Gateway v2 (payload version 2.0) headers are lowercased.
    headers = {k.lower(): v for k, v in event.get('headers', {}).items()}
    twilio_signature = headers.get('x-twilio-signature')
    
    # Reconstruct the full request URL
    # Example: https://{api_id}.execute-api.{region}.amazonaws.com/{stage}/your-webhook-path
    request_url = f"https://{headers.get('host')}{event.get('rawPath')}"
    if event.get('rawQueryString'):
        request_url += f"?{event['rawQueryString']}"

    # Reconstruct the POST parameters
    post_vars = {}
    if event.get('body'):
        body_str = event['body']
        if event.get('isBase64Encoded', False):
            try:
                body_str = base64.b64decode(body_str).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to decode base64 body for validation: {e}")
                return {'statusCode': 400, 'body': 'Bad Request: Invalid body encoding.'}
        post_vars = urllib.parse.parse_qs(body_str)
        # urllib.parse.parse_qs returns lists for values, convert to single values
        post_vars = {k: v[0] for k, v in post_vars.items()}

    if not validator.validate(request_url, post_vars, twilio_signature):
        logger.warning(f"Twilio request validation failed for URL: {request_url}, Signature: {twilio_signature}")
        return {'statusCode': 403, 'body': 'Forbidden: Invalid Twilio Signature.'}
    logger.info("Twilio request validation successful.")
    
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
        "Identification started! \n\n(Please note: By sending a picture to the Little Tree Farms Horse ID number, "
        "you consent to receive a response. Respond STOP to stop receiving messages.)"
    )
    twilio_response_body = f'<Response><Message>{response_message}</Message></Response>'

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/xml'},
        'body': twilio_response_body
    }