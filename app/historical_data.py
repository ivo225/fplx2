from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import shutil

class HistoricalDataManager:
    """Manages storage and retrieval of historical FPL data."""
    
    def __init__(self):
        """Initialize the historical data manager."""
        self.base_path = Path('data/historical')
        self.sentiment_path = self.base_path / 'sentiment'
        self.predictions_path = self.base_path / 'predictions'
        self.video_analysis_path = self.sentiment_path / 'video_analysis'
        self.weekly_scores_path = self.sentiment_path / 'weekly_scores'
        self.model_predictions_path = self.predictions_path / 'model_predictions'
        
        # Create directory structure if it doesn't exist
        for path in [self.sentiment_path, self.video_analysis_path, 
                    self.weekly_scores_path, self.predictions_path,
                    self.model_predictions_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_sentiment_scores(self, gameweek: int, sentiment_data: Dict) -> None:
        """Store sentiment analysis results for a specific gameweek.
        
        Args:
            gameweek: The gameweek number
            sentiment_data: Dictionary containing sentiment scores and metadata
        """
        timestamp = datetime.utcnow().isoformat()
        data_to_save = {
            'timestamp': timestamp,
            'gameweek': gameweek,
            'data': sentiment_data
        }
        
        file_path = self.weekly_scores_path / f'gw{gameweek}.json'
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        # Also save to archive with timestamp for historical tracking
        archive_path = self.weekly_scores_path / 'archive' / f'gw{gameweek}_{timestamp}.json'
        archive_path.parent.mkdir(exist_ok=True)
        with open(archive_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def save_video_analysis(self, gameweek: int, video_data: List[Dict]) -> None:
        """Store video analysis results for a specific gameweek.
        
        Args:
            gameweek: The gameweek number
            video_data: List of dictionaries containing video analysis results
        """
        timestamp = datetime.utcnow().isoformat()
        data_to_save = {
            'timestamp': timestamp,
            'gameweek': gameweek,
            'videos': video_data
        }
        
        file_path = self.video_analysis_path / f'gw{gameweek}.json'
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def save_model_predictions(self, gameweek: int, predictions: Dict, 
                             model_info: Optional[Dict] = None) -> None:
        """Store model predictions for a specific gameweek.
        
        Args:
            gameweek: The gameweek number
            predictions: Dictionary of player predictions
            model_info: Optional metadata about the model used
        """
        timestamp = datetime.utcnow().isoformat()
        data_to_save = {
            'timestamp': timestamp,
            'gameweek': gameweek,
            'predictions': predictions,
            'model_info': model_info or {}
        }
        
        file_path = self.model_predictions_path / f'gw{gameweek}.json'
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def get_sentiment_scores(self, gameweek: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Retrieve sentiment scores for a specific gameweek or all gameweeks.
        
        Args:
            gameweek: Optional gameweek number. If None, returns all gameweeks.
        
        Returns:
            Dictionary or list of dictionaries containing sentiment scores
        """
        if gameweek is not None:
            file_path = self.weekly_scores_path / f'gw{gameweek}.json'
            if not file_path.exists():
                return {}
            with open(file_path, 'r') as f:
                return json.load(f)
        
        # Return all gameweeks
        scores = []
        for file_path in sorted(self.weekly_scores_path.glob('gw*.json')):
            if 'archive' not in str(file_path):  # Skip archive files
                with open(file_path, 'r') as f:
                    scores.append(json.load(f))
        return scores
    
    def get_video_analysis(self, gameweek: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Retrieve video analysis for a specific gameweek or all gameweeks.
        
        Args:
            gameweek: Optional gameweek number. If None, returns all gameweeks.
        
        Returns:
            Dictionary or list of dictionaries containing video analysis data
        """
        if gameweek is not None:
            file_path = self.video_analysis_path / f'gw{gameweek}.json'
            if not file_path.exists():
                return {}
            with open(file_path, 'r') as f:
                return json.load(f)
        
        # Return all gameweeks
        analyses = []
        for file_path in sorted(self.video_analysis_path.glob('gw*.json')):
            with open(file_path, 'r') as f:
                analyses.append(json.load(f))
        return analyses
    
    def get_model_predictions(self, gameweek: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Retrieve model predictions for a specific gameweek or all gameweeks.
        
        Args:
            gameweek: Optional gameweek number. If None, returns all gameweeks.
        
        Returns:
            Dictionary or list of dictionaries containing model predictions
        """
        if gameweek is not None:
            file_path = self.model_predictions_path / f'gw{gameweek}.json'
            if not file_path.exists():
                return {}
            with open(file_path, 'r') as f:
                return json.load(f)
        
        # Return all gameweeks
        predictions = []
        for file_path in sorted(self.model_predictions_path.glob('gw*.json')):
            with open(file_path, 'r') as f:
                predictions.append(json.load(f))
        return predictions
    
    def get_historical_trends(self, player_name: str) -> Dict:
        """Get historical trends for a specific player.
        
        Args:
            player_name: Name of the player to get trends for
        
        Returns:
            Dictionary containing historical trends for the player
        """
        trends = {
            'sentiment_scores': [],
            'predictions': [],
            'gameweeks': []
        }
        
        # Collect sentiment scores
        for score_data in self.get_sentiment_scores():
            gw = score_data['gameweek']
            if player_name in score_data['data']['sentiment_scores']:
                trends['sentiment_scores'].append(
                    score_data['data']['sentiment_scores'][player_name]
                )
                trends['gameweeks'].append(gw)
        
        # Collect predictions
        for pred_data in self.get_model_predictions():
            gw = pred_data['gameweek']
            if player_name in pred_data['predictions']:
                trends['predictions'].append(
                    pred_data['predictions'][player_name]
                )
        
        return trends
    
    def backup_historical_data(self, backup_dir: Optional[str] = None) -> None:
        """Create a backup of all historical data.
        
        Args:
            backup_dir: Optional directory to store backup. If None, uses 'data/backups'
        """
        if backup_dir is None:
            backup_dir = 'data/backups'
        
        backup_path = Path(backup_dir)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_path / f'historical_backup_{timestamp}'
        
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all historical data
        shutil.copytree(self.base_path, backup_path / 'historical', dirs_exist_ok=True)
        
        print(f"Backup created at: {backup_path}") 