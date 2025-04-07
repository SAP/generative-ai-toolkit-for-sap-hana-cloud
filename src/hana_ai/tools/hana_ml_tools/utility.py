"""
Utility functions for the HANA ML tools.
"""
import json
from datetime import datetime, date
from pandas import Timestamp
from numpy import int64

class _CustomEncoder(json.JSONEncoder):
    """
    This class is used to encode the model attributes into JSON string.
    """
    def default(self, obj): #pylint: disable=arguments-renamed
        if isinstance(obj, (Timestamp, datetime, date)):
            # Convert Timestamp, datetime or date to ISO string
            return obj.isoformat()
        elif isinstance(obj, (int64, int)):
            # Convert numpy int64 or Python int to Python int
            return int(obj)
        # Let other types use the default handler
        return super().default(obj)
