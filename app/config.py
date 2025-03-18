import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# YouTube API configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# FPL API configuration
FPL_API_URL = "https://fantasy.premierleague.com/api"

# Sentiment analysis configuration
SENTIMENT_LOOKBACK_DAYS = 3  # Number of days to look back for videos
MAX_COMMENTS_PER_VIDEO = 100  # Maximum number of comments to analyze per video

# Model configuration
MODEL_PATH = "models/xgb_model.json"
SCALER_PATH = "models/standard_scaler.pkl"

# Optimization configuration
TOTAL_BUDGET = 100.0
MIN_PLAYERS_PER_TEAM = 0
MAX_PLAYERS_PER_TEAM = 3 