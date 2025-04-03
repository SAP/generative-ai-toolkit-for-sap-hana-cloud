import os
import hana_ai


"""
hana_ai is a Python package for AI/ML related utilities.
"""

# The root directory of the package
root_dir = os.path.dirname(hana_ai.__file__)

version_file = os.path.join(root_dir, "..", "..", "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()
