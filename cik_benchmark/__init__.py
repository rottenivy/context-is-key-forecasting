import os

env = os.getenv("MY_ENV", "dev")  # Default to "dev"

if env == "dev":
    from .init_dev import *
else:
    from .init_cik import *
