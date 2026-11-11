import unittest
from unittest.mock import patch

from mlf_assistant.security import (
    _verify_password,
    authenticate_user,
    get_active_user_role,
    hash_admin_password,
    session_is_authorized,
)


class AdminPasswordTests(unittest.TestCase):
    def test_hash_verifies_without_persisting_plaintext(self):
        password = "Ministry-Access-2026!"
        encoded = hash_admin_password(password)
        self.assertNotIn(password, encoded)
        self.assertTrue(_verify_password(password, "", encoded))
        self.assertFalse(_verify_password("wrong-password", "", encoded))

    def test_short_password_is_rejected(self):
        with self.assertRaises(ValueError):
            hash_admin_password("short")

    def test_account_authentication_rejects_inactive_accounts(self):
        account = {
            "username": "officer",
            "password_hash": hash_admin_password("Officer-Access-2026!"),
            "role": "officer",
            "active": False,
        }
        with patch("mlf_assistant.db.get_account_auth", return_value=account):
            self.assertIsNone(get_active_user_role("officer"))

    def test_configured_admin_does_not_fall_back_to_stale_database_password(self):
        settings = type(
            "Settings",
            (),
            {
                "admin_username": "admin",
                "admin_password": "",
                "admin_password_hash": hash_admin_password("Current-Admin-2026!"),
            },
        )()
        with (
            patch("mlf_assistant.security.get_settings", return_value=settings),
            patch("mlf_assistant.db.get_account_auth") as get_account_auth,
        ):
            self.assertEqual(authenticate_user("admin", "Current-Admin-2026!"), "admin")
            self.assertIsNone(authenticate_user("admin", "Old-Admin-Password!"))
            get_account_auth.assert_not_called()

    def test_authenticated_session_requires_active_matching_role(self):
        with (
            patch("mlf_assistant.security.time.time", return_value=1500),
            patch("mlf_assistant.security.get_active_user_role", return_value="officer"),
        ):
            self.assertTrue(session_is_authorized("officer", "officer", 1000))
            self.assertFalse(session_is_authorized("officer", "admin", 1000))

    def test_authenticated_session_expires_and_rejects_future_timestamps(self):
        with (
            patch("mlf_assistant.security.time.time", return_value=2000),
            patch("mlf_assistant.security.get_active_user_role") as get_role,
        ):
            self.assertFalse(session_is_authorized("officer", "officer", 0))
            self.assertFalse(session_is_authorized("officer", "officer", 2001))
            get_role.assert_not_called()


if __name__ == "__main__":
    unittest.main()