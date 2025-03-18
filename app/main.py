from data_fetcher import fetch_and_process_data, FPLDataFetcher
from model_trainer import FPLModelTrainer
from team_optimizer import TeamOptimizer
from pathlib import Path
import pandas as pd
from config import YOUTUBE_API_KEY
import argparse

def get_prediction_gameweek(current_gw):
    """Determine the gameweek to predict for based on current gameweek"""
    if current_gw is None:
        return None
    return current_gw + 1 if current_gw < 38 else None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FPL Team Optimizer with Sentiment Analysis')
    parser.add_argument('--force-refresh-sentiment', action='store_true',
                       help='Force recalculation of sentiment scores ignoring cache')
    args = parser.parse_args()
    
    print("FPL Team Optimizer with Sentiment Analysis")
    print("=========================================")
    
    if args.force_refresh_sentiment:
        print("\n⚠️ Force refreshing sentiment scores - ignoring cache")
    
    prediction_gw = None
    
    # Check if YouTube API key is configured
    if not YOUTUBE_API_KEY:
        print("\nWarning: YouTube API key not found in environment variables.")
        print("Sentiment analysis will be disabled.")
        print("To enable sentiment analysis, please:")
        print("1. Copy .env.template to .env")
        print("2. Add your YouTube API key to the .env file")
    
    # Check if we have cached data
    data_path = Path('data/fpl_data.csv')
    if data_path.exists():
        print("\nLoading cached FPL data...")
        players_df = pd.read_csv(data_path)
        # Fetch current gameweek from API
        try:
            raw_data = FPLDataFetcher.fetch_bootstrap_static()
            current_gw = FPLDataFetcher.get_current_gameweek(raw_data)
            prediction_gw = get_prediction_gameweek(current_gw)
            if prediction_gw:
                print(f"\n🎯 Predicting team for Gameweek {prediction_gw}")
                print(f"(Current Gameweek: {current_gw})")
            else:
                print("\nWarning: Could not determine prediction gameweek")
        except:
            print("\nWarning: Could not fetch current gameweek information")
    else:
        print("\nFetching fresh FPL data...")
        raw_data = FPLDataFetcher.fetch_bootstrap_static()
        fixtures_data = FPLDataFetcher.fetch_fixtures()
        
        # Get current gameweek
        current_gw = FPLDataFetcher.get_current_gameweek(raw_data)
        prediction_gw = get_prediction_gameweek(current_gw)
        if prediction_gw:
            print(f"\n🎯 Predicting team for Gameweek {prediction_gw}")
            print(f"(Current Gameweek: {current_gw})")
        else:
            print("\nWarning: Could not determine prediction gameweek")
        
        # Calculate fixture difficulty
        fixture_difficulty = FPLDataFetcher.calculate_fixture_difficulty(fixtures_data)
        
        # Process player data
        players_df = FPLDataFetcher.process_player_data(raw_data, fixture_difficulty)
        if players_df is None:
            print("Failed to fetch FPL data. Please check your internet connection.")
            return
    
    # Check if we have a trained model
    model_path = Path('models/xgboost_model.pkl')
    if model_path.exists():
        print("\nLoading existing model...")
        trainer = FPLModelTrainer.load_model()
    else:
        print("\nTraining new model...")
        trainer = FPLModelTrainer()
        trainer.train_model(players_df)
    
    # Predict points for all players
    print("\nPredicting player points...")
    predicted_points = trainer.predict_points(players_df)
    
    # Optimize team
    print("\nOptimizing team selection...")
    optimizer = TeamOptimizer(
        players_df,
        predicted_points,
        target_gameweek=prediction_gw,
        force_refresh_sentiment=args.force_refresh_sentiment
    )
    optimal_team = optimizer.optimize_team()
    
    # Display results
    print("\nOptimal Team Selection:")
    print("======================")
    
    # Set display options for better formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    
    # Display team statistics
    print("\nTeam Statistics:")
    print(f"Total Predicted Points: {optimal_team['predicted_points'].sum():.2f}")
    print(f"Total Cost: £{optimal_team['now_cost'].sum():.1f}M")
    print(f"Remaining Budget: £{0.0:.1f}M")
    if 'difficulty' in optimal_team.columns:
        print(f"Average Fixture Difficulty: {optimal_team['difficulty'].mean():.2f} (1=Easy, 5=Hard)")
    print(f"Average Starting XI Sentiment: {optimal_team[optimal_team['status'].str.contains('Starting')]['sentiment_score'].mean():.2f}")
    
    # Display excluded players
    if 'is_available' in players_df.columns:
        excluded_players = players_df[~players_df['is_available']].sort_values(['team', 'web_name'])
        print("\nExcluded players due to injuries/availability:")
        for _, player in excluded_players.iterrows():
            print(f"{player['web_name']} ({player['team']}) - {player.get('status', 'u')}")
    else:
        print("\nPlayer availability information not available")
    
    # Display players in excellent form
    high_form_players = players_df[players_df['form'] > 5.0].sort_values('form', ascending=False)
    print("\nPlayers in excellent form (form > 5.0):")
    for _, player in high_form_players.iterrows():
        print(f"{player['web_name']} ({player['team']}) - Form: {player['form']}")
    
    # Add a blank line between positions for better readability
    print("\nSelected Squad by Position:")
    print("=========================")
    position_groups = optimal_team.groupby('position')
    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        if position in position_groups.groups:
            position_df = position_groups.get_group(position)
            print(f"\n{position}:")
            print(position_df.to_string(index=False))

if __name__ == "__main__":
    main() 