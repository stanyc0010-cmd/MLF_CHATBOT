import os
import unittest
from unittest.mock import patch

from mlf_assistant.config import get_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()

    def test_reads_streamlit_secrets(self):
        with (
            patch(
                "mlf_assistant.config._streamlit_secret_values",
                return_value={
                    "database_url": "postgresql+psycopg://cloud/database",
                    "llm_api_key": "cloud-key",
                },
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = get_settings()

        self.assertEqual(settings.database_url, "postgresql+psycopg://cloud/database")
        self.assertEqual(settings.llm_api_key, "cloud-key")

    def test_environment_variables_override_streamlit_secrets(self):
        with (
            patch(
                "mlf_assistant.config._streamlit_secret_values",
                return_value={"llm_api_key": "cloud-key"},
            ),
            patch.dict(os.environ, {"MLF_LLM_API_KEY": "environment-key"}, clear=True),
        ):
            settings = get_settings()

        self.assertEqual(settings.llm_api_key, "environment-key")