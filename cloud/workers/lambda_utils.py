"""Shared utilities for invoking Lambda from within Lambda (self-chaining)."""

import json
import os

import boto3

FUNCTION_NAME = os.environ.get("ML_WORKER_LAMBDA_NAME", "horse-id-ml-worker")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

_client = None


def get_lambda_client():
    global _client
    if _client is None:
        _client = boto3.client("lambda", region_name=REGION)
    return _client


def invoke_lambda(task: str, payload: dict):
    """Invoke the ML worker Lambda asynchronously with the given task and payload."""
    client = get_lambda_client()
    event = {"task": task, **payload}
    response = client.invoke(
        FunctionName=FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps(event),
    )
    status = response.get("StatusCode", 0)
    if status != 202:
        print(f"[lambda_utils] Unexpected status {status} invoking {task}")
    return status


def invoke_detection(photos: list[dict]):
    """Invoke detection Lambda for a batch of photos."""
    invoke_lambda("detect", {"photos": photos})


def invoke_extraction(photos: list[dict]):
    """Invoke extraction Lambda for a batch of photos."""
    invoke_lambda("extract", {"photos": photos})
