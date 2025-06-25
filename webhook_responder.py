import json
import os
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def webhook_handler(event, context):
    """
    Receives the initial webhook from Twilio, invokes the processor asynchronously,
    and returns an immediate response to Twilio.
    This should be the handler for your 'webhook-responder' Lambda.
    """
    logger.info("--- webhook_handler invoked ---")
    
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