import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt


JWT_ALGORITHM = "HS256"
TOKEN_HOURS = 24 * 14


def _secret():
    return os.getenv("INFINITYAI_JWT_SECRET") or os.getenv("INFINITYAI_API_KEY", "default-key")


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return "pbkdf2_sha256$120000${}${}".format(
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_password(password, stored_hash):
    try:
        _name, rounds, salt_b64, digest_b64 = stored_hash.split("$", 3)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(rounds))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_token(user_id):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def decode_token(token):
    payload = jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])
    return int(payload["sub"])
