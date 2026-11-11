import hashlib
import hmac
import secrets
import time

from mlf_assistant.config import get_settings


class AuthorizationError(Exception):
    """Raised when an administrative operation is attempted without authorization."""


def verify_admin(username: str, password: str) -> bool:
    return authenticate_user(username, password) == "admin"


def authenticate_user(username: str, password: str) -> str | None:
    settings = get_settings()
    if hmac.compare_digest(username, settings.admin_username) and (
        settings.admin_password or settings.admin_password_hash
    ):
        return (
            "admin"
            if _verify_password(password, settings.admin_password, settings.admin_password_hash)
            else None
        )

    from mlf_assistant.db import get_account_auth

    account = get_account_auth(username)
    if not account or not account["active"] or not _verify_password(password, "", account["password_hash"]):
        return None
    return account["role"]


def get_active_user_role(username: str) -> str | None:
    settings = get_settings()
    if (
        hmac.compare_digest(username, settings.admin_username)
        and (settings.admin_password or settings.admin_password_hash)
    ):
        return "admin"

    from mlf_assistant.db import get_account_auth

    account = get_account_auth(username)
    if not account or not account["active"]:
        return None
    return account["role"]


def session_is_authorized(username: object, role: object, authenticated_at: object) -> bool:
    if (
        not isinstance(username, str)
        or role not in {"admin", "officer"}
        or not isinstance(authenticated_at, (int, float))
        or not 0 <= time.time() - authenticated_at < 1800
    ):
        return False
    return get_active_user_role(username) == role


def hash_admin_password(password: str, *, rounds: int = 600_000) -> str:
    if len(password) < 14:
        raise ValueError("Administrator passwords must contain at least 14 characters.")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), rounds).hex()
    return f"pbkdf2_sha256${rounds}${salt}${digest}"


def _verify_password(password: str, plain: str, encoded: str) -> bool:
    if encoded:
        try:
            algorithm, rounds, salt, expected = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), int(rounds)
            ).hex()
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False
    return bool(plain) and hmac.compare_digest(password, plain)


def require_admin(is_admin: bool) -> None:
    if not is_admin:
        raise AuthorizationError("Administrator authorization is required.")