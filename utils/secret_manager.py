import os
import sys
import boto3
import json
from botocore.exceptions import ClientError
from pathlib import Path

def export_key_to_os_environ(secret_id, dry_run=None, mode=None):
    secret_key = get_secret_key(secret_id)

    if not secret_key:
        if dry_run:
            print("ℹ️  DRY-RUN mode → continuing without Anthropic key")
            return   # ← IMPORTANT: skip assignment and exit function
        elif mode == "sagemaker":
            print("ℹ️  SageMaker mode without key → some features will be skipped")
            return   # ← also safe to skip in sagemaker if key is optional
        else:
            print(f"❌ {secret_id} not found in Secrets Manager / env / .env")
            print(f"   Local dev → create .env with {secret_id}=sk-ant-...")
            print("   SageMaker → store key in Secrets Manager or set env var")
            sys.exit(1)

    # Only reach here if we actually have a key
    os.environ[secret_id] = secret_key
    print(f"✓ Exported {secret_id} to os.environ")

    # Optional safety check
    if not os.getenv(secret_id):
        print(f"Warning: {secret_id} not readable from os.environ after assignment")
        
def get_secret_key(secret_id) -> str | None:
    """
    Load API key from (in order):
    1. AWS Secrets Manager (preferred in SageMaker / AWS)
    2. Environment variable
    3. .env file (local dev only)
    Returns key or None → caller should decide whether to raise or continue
    """
    # ── 1. AWS Secrets Manager ────────────────────────────────────────
    if os.getenv("AWS_EXECUTION_ENV") or "sagemaker" in os.getenv("PATH", "").lower():
        # In SageMaker / AWS → try Secrets Manager first
        try:
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=secret_id)

            if "SecretString" in response:
                secret = response["SecretString"]
                # Two common formats
                if secret.startswith("{"):
                    data = json.loads(secret)
                    key = data.get(secret_id) or data.get("api_key") or next(iter(data.values()), None)
                else:
                    key = secret.strip()

                if key and key.startswith("sk-ant-"):
                    print(f"✓ Key loaded from AWS Secrets Manager (length: {len(key)})")
                    return key
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"Secret {secret_id} not found in Secrets Manager")
            elif e.response["Error"]["Code"] == "AccessDeniedException":
                print("IAM role lacks secretsmanager:GetSecretValue permission")
            else:
                print(f"Secrets Manager error: {e}")

    # ── 2. Environment variable ───────────────────────────────────────
    key = os.getenv(secret_id)
    if key and key.startswith("sk-ant-"):
        print(f"✓ Key loaded from environment variable (length: {len(key)})")
        return key

    # ── 3. .env file (local development only) ─────────────────────────
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            key = os.getenv(secret_id)
            if key and key.startswith("sk-ant-"):
                print(f"✓ Key loaded from .env file (length: {len(key)})")
                return key
    except ImportError:
        pass  # dotenv not installed → skip silently

    return None