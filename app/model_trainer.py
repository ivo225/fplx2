import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import pickle
from pathlib import Path

class FPLModelTrainer:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        # Features to use for prediction
        feature_columns = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
            'bonus', 'bps', 'influence', 'creativity', 'threat',
            'selected_by_percent', 'form', 'value_season',
            'fixture_difficulty'  # Add fixture difficulty as a feature
        ]
        
        # Target variable
        target = df['total_points']
        
        # Convert string values to numeric if needed
        features = df[feature_columns].copy()
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Fill missing values
        features = features.fillna(0)
        
        # Adjust predicted points based on fixture difficulty
        # Lower difficulty (easier fixtures) should increase predicted points
        features['fixture_adjusted_form'] = features['form'].astype(float) * (5 - features['fixture_difficulty'])
        
        return features, target
    
    def train_model(self, df: pd.DataFrame) -> None:
        """Train the XGBoost model."""
        X, y = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        self._save_model()
        
        # Print feature importance
        self._print_feature_importance(X.columns)
    
    def predict_points(self, player_data: pd.DataFrame) -> np.ndarray:
        """Predict points for players."""
        X, _ = self.prepare_features(player_data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Adjust predictions based on fixture difficulty
        fixture_adjustment = (5 - player_data['fixture_difficulty']) * 0.1
        adjusted_predictions = predictions * (1 + fixture_adjustment)
        
        return adjusted_predictions
    
    def _print_feature_importance(self, feature_names):
        """Print feature importance scores."""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print("==================")
        print(feature_importance.to_string(index=False))
    
    def _save_model(self):
        """Save the trained model and scaler."""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @staticmethod
    def load_model():
        """Load a trained model and scaler."""
        models_dir = Path('models')
        
        with open(models_dir / 'xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        trainer = FPLModelTrainer()
        trainer.model = model
        trainer.scaler = scaler
        return trainer 