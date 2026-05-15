#!/usr/bin/env python3
"""Unit: Parse VCAP_SERVICES from an env-style file.

This test reads the committed example file at nutest/testscripts/env.example and
asserts we can derive ConnectionContext parameters from it.

It intentionally does NOT create a hana_ml.ConnectionContext (no network).
"""

from __future__ import annotations

import os
import unittest


class TestVCAPServicesFileParsing(unittest.TestCase):
    def test_build_cc_params_from_vcap_services_file(self):
        from hana_ai.tools import toolkit as tk_mod

        here = os.path.dirname(__file__)
        env_example = os.path.normpath(os.path.join(here, "..", "env.example"))
        self.assertTrue(os.path.isfile(env_example), f"Missing test fixture file: {env_example}")

        params = tk_mod._build_cc_params_from_vcap_services_file(env_example)
        self.assertIsInstance(params, dict)

        # Basic required fields
        self.assertIn("address", params)
        self.assertIn("port", params)
        self.assertIn("user", params)
        self.assertIn("password", params)

        self.assertIsInstance(params["address"], str)
        self.assertTrue(params["address"].endswith("hanacloud.ondemand.com"))
        self.assertEqual(params["port"], 443)
        self.assertIsInstance(params["user"], str)
        self.assertTrue(len(params["user"]) > 5)
        self.assertIsInstance(params["password"], str)
        self.assertTrue(len(params["password"]) > 10)

        # TLS flags derived from JDBC URL query
        self.assertEqual(params.get("encrypt"), True)
        self.assertEqual(params.get("sslValidateCertificate"), True)


if __name__ == "__main__":
    unittest.main()
