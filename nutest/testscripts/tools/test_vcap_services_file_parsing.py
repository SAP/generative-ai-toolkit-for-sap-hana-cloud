#!/usr/bin/env python3
"""Unit tests for environment-based ConnectionContext parameters."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestVCAPServicesFileParsing(unittest.TestCase):
    def test_build_cc_params_from_userkey_env(self):
        from hana_ai.tools import toolkit as tk_mod
        with patch.dict(
            "os.environ",
            {"HANA_USERKEY": "XinKey2", "HANA_SSL_VALIDATE": "true"},
            clear=True,
        ):
            params = tk_mod._build_cc_params_from_env()
        self.assertEqual(params, {
            "userkey": "XinKey2",
            "sslValidateCertificate": True,
        })


if __name__ == "__main__":
    unittest.main()
