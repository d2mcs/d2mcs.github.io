"""This module contains code for loading various event-specific
configuration data from the data folder.
"""

import json
from datetime import datetime

class EventConf:
    """Event configuration class. Currently this just loads a few fields
    stored in a JSON file to member variables

    Parameters
    ----------
    conf_path : str
        Path to configuration file.
    """
    def __init__(self, conf_path):
        with open(conf_path) as conf_f:
            conf_json = json.load(conf_f)
        self.tabs = conf_json["tabs"]
        self.name = conf_json["name"]
        self.start = datetime.fromisoformat(conf_json["start_date"]
                            ).timestamp()
        self.end = datetime.fromisoformat(conf_json["end_date"]).timestamp()
