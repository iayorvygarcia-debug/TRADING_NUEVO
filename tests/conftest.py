"""Global pytest fixtures and environment configuration."""
import os

# During unit tests, disable operational capital override so standard test fixtures (e.g. 100k equity) execute cleanly
os.environ["USE_OPERATIONAL_CAPITAL"] = "false"
