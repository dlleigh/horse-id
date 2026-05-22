#!/usr/bin/env bash
# Sets up an EventBridge scheduled rule to ping the ML worker Lambda every 10 minutes
# during active hours (6 AM - 10 PM ET) to avoid cold starts.
#
# Also bumps Lambda memory from 4096 to 5120 MB for headroom.
#
# Usage: ./setup-warmup-ping.sh [--profile PROFILE] [--region REGION]
#
# To remove: ./setup-warmup-ping.sh --teardown

set -euo pipefail

FUNCTION_NAME="horse-id-ml-worker"
RULE_NAME="horse-id-ml-worker-warmup"
REGION="us-east-2"
PROFILE=""
TEARDOWN=false
MEMORY_SIZE=5120

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="--profile $2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --teardown) TEARDOWN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

AWS="aws $PROFILE --region $REGION"
FUNCTION_ARN="arn:aws:lambda:${REGION}:685175429625:function:${FUNCTION_NAME}"

if $TEARDOWN; then
    echo "Removing warmup ping rule..."
    $AWS events remove-targets --rule "$RULE_NAME" --ids "warmup-target" 2>/dev/null || true
    $AWS events delete-rule --name "$RULE_NAME" 2>/dev/null || true
    $AWS lambda remove-permission --function-name "$FUNCTION_NAME" --statement-id "warmup-ping-permission" 2>/dev/null || true
    echo "Done. Rule removed."
    exit 0
fi

# 1. Bump Lambda memory
echo "Updating Lambda memory to ${MEMORY_SIZE} MB..."
$AWS lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --memory-size "$MEMORY_SIZE" \
    --output text --query 'MemorySize'
echo "Memory updated."

# 2. Create EventBridge rule — every 10 minutes, 6 AM–10 PM ET (10:00–02:00 UTC)
echo "Creating EventBridge rule: ${RULE_NAME}..."
$AWS events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "cron(0/10 10-23,0-1 ? * * *)" \
    --state ENABLED \
    --description "Ping ML worker every 10 min during 6AM-10PM ET to prevent cold starts" \
    --output text --query 'RuleArn'

# 3. Grant EventBridge permission to invoke the Lambda
echo "Adding Lambda invoke permission for EventBridge..."
$AWS lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "warmup-ping-permission" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:${REGION}:685175429625:rule/${RULE_NAME}" \
    2>/dev/null || echo "Permission already exists, skipping."

# 4. Add the Lambda as a target with the ping payload
echo "Setting Lambda as rule target..."
$AWS events put-targets \
    --rule "$RULE_NAME" \
    --targets "[{\"Id\": \"warmup-target\", \"Arn\": \"${FUNCTION_ARN}\", \"Input\": \"{\\\"task\\\": \\\"ping\\\"}\"}]" \
    --output text

echo ""
echo "Setup complete. The ML worker will be pinged every 10 minutes from 6 AM to 10 PM ET."
echo "To remove: $0 --teardown ${PROFILE:+$PROFILE}"
