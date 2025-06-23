#!/bin/sh

# Define the mock event JSON
MOCK_TWILIO_EVENT_JSON='{
  "SmsMessageSid": "SMxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "AccountSid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "From": "+1234567890",
  "To": "+1987654321",
  "Body": "Here is a horse image for identification.",
  "NumMedia": "1",
  "MediaUrl0": "http://host.containers.internal:5000",
  "MediaContentType0": "image/jpeg"
}'

# Start the container in the background, mapping port 8080
echo "Starting Lambda container in background..."
podman run -d --rm -p 8080:8080 \
  -v ~/.aws:/root/.aws \
  -e HORSE_ID_DATA_ROOT="/data" \
  -e AWS_PROFILE=${AWS_PROFILE} \
  --name horse-id-lambda-container \
  horse-id-lambda-image

# Wait a moment for the container to start up (adjust if needed)
sleep 5

# Invoke the Lambda function via HTTP POST
echo "Invoking Lambda function..."
curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d "$MOCK_TWILIO_EVENT_JSON"

# Stop and remove the container
echo "Stopping Lambda container..."
podman stop horse-id-lambda-container