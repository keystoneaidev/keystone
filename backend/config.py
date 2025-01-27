import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration settings."""
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///keystone.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
    API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "100/hour")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "models/checkpoints/")
    BLOCKCHAIN_PROVIDER = os.getenv("BLOCKCHAIN_PROVIDER", "https://mainnet.infura.io/v3/YOUR_INFURA_KEY")
    
    @staticmethod
    def load_config_from_json(file_path: str):
        """Load configuration values from a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as config_file:
                return json.load(config_file)
        return {}

# Load additional configurations from JSON file if available
CONFIG_FILE_PATH = "keystone_config.json"
extra_config = Config.load_config_from_json(CONFIG_FILE_PATH)

# Merge JSON configurations with environment-based defaults
for key, value in extra_config.items():
    setattr(Config, key.upper(), value)

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("keystone")
logger.info("Configuration loaded successfully.")

