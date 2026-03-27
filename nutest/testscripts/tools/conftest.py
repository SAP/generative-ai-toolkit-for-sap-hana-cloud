import os
import sys


def pytest_configure():
    """Ensure legacy tests can import shared base classes.

Many tests under nutest/testscripts/tools import `testML_BaseTestClass` as a top-level
module. When running pytest from repo root, we need to add nutest/testscripts to sys.path.
"""

    here = os.path.dirname(__file__)
    testscripts_dir = os.path.abspath(os.path.join(here, ".."))
    if testscripts_dir not in sys.path:
        sys.path.insert(0, testscripts_dir)
