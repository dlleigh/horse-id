"""Resolve configuration from SSM Parameter Store (Lambda) or .env (local).

On Lambda, secrets are fetched from SSM at cold start and injected into os.environ.
Locally, dotenv loads from cloud/.env as before.
"""

import os

SSM_PREFIX = "/horse-id/"

# Map SSM parameter names -> env var names
SSM_PARAMS = {
    "database-url": "DATABASE_URL",
    "drive-service-account-key": "GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY",
    "twilio-account-sid": "TWILIO_ACCOUNT_SID",
    "twilio-auth-token": "TWILIO_AUTH_TOKEN",
}


def _load_from_ssm():
    """Fetch secrets from SSM and set as env vars."""
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    ssm = boto3.client("ssm", region_name=region)

    for param_name, env_var in SSM_PARAMS.items():
        if os.environ.get(env_var):
            continue  # already set (e.g. by Lambda env config)
        try:
            resp = ssm.get_parameter(
                Name=f"{SSM_PREFIX}{param_name}", WithDecryption=True
            )
            os.environ[env_var] = resp["Parameter"]["Value"]
        except Exception as e:
            print(f"Warning: could not fetch SSM param {param_name}: {e}")


def _load_from_dotenv():
    """Load .env file for local development."""
    try:
        from dotenv import load_dotenv

        env_path = os.path.join(os.path.dirname(__file__), "../.env")
        load_dotenv(env_path)
    except ImportError:
        pass  # dotenv not installed, rely on env vars


def init():
    """Initialize configuration. Call once at startup."""
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        # Running in Lambda — use SSM
        _load_from_ssm()
    else:
        # Local dev — use .env
        _load_from_dotenv()
