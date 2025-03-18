import requests
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

class FPLDataFetcher:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    @staticmethod
    def fetch_bootstrap_static() -> Dict[str, Any]:
        """Fetch basic FPL data including players, teams, and game rules."""
        response = requests.get(f"{FPLDataFetcher.BASE_URL}/bootstrap-static/")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_current_gameweek(data: Dict[str, Any]) -> int:
        """Get the current gameweek number."""
        events = data.get('events', [])
        for event in events:
            if event.get('is_current'):
                return event.get('id')
        return None
    
    @staticmethod
    def fetch_fixtures() -> List[Dict[str, Any]]:
        """Fetch all fixtures for the season."""
        response = requests.get(f"{FPLDataFetcher.BASE_URL}/fixtures/")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def calculate_fixture_difficulty(fixtures: List[Dict[str, Any]], num_future_games: int = 3) -> pd.DataFrame:
        """Calculate fixture difficulty for each team for the next N games."""
        # Convert fixtures to DataFrame
        fixtures_df = pd.DataFrame(fixtures)
        
        # Filter only future fixtures
        future_fixtures = fixtures_df[fixtures_df['finished'] == False].copy()
        
        # Create team difficulty mappings
        team_difficulties = {}
        
        # Calculate average difficulty for each team's next N games
        for team_id in range(1, 21):  # Assuming 20 teams in the league
            # Get team's home games
            home_games = future_fixtures[future_fixtures['team_h'] == team_id][['team_a', 'team_h_difficulty']]
            home_games = home_games.head(num_future_games)
            
            # Get team's away games
            away_games = future_fixtures[future_fixtures['team_a'] == team_id][['team_h', 'team_a_difficulty']]
            away_games = away_games.head(num_future_games)
            
            # Combine and calculate average difficulty
            total_games = len(home_games) + len(away_games)
            if total_games > 0:
                avg_difficulty = (
                    home_games['team_h_difficulty'].sum() + 
                    away_games['team_a_difficulty'].sum()
                ) / total_games
                team_difficulties[team_id] = avg_difficulty
            else:
                team_difficulties[team_id] = 3.0  # Default medium difficulty
        
        return pd.DataFrame(list(team_difficulties.items()), 
                          columns=['team', 'fixture_difficulty'])
    
    @staticmethod
    def process_player_data(data: Dict[str, Any], fixture_difficulty: pd.DataFrame) -> pd.DataFrame:
        """Process raw FPL data into a pandas DataFrame with relevant features."""
        players_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        
        # Select relevant features for prediction
        relevant_features = [
            'id', 'web_name', 'team', 'element_type',
            'now_cost', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'own_goals',
            'penalties_saved', 'penalties_missed', 'yellow_cards',
            'red_cards', 'saves', 'bonus', 'bps', 'influence',
            'creativity', 'threat', 'ict_index', 'total_points',
            'selected_by_percent', 'form', 'points_per_game',
            'value_season', 'transfers_in', 'transfers_out',
            'status', 'chance_of_playing_next_round'
        ]
        
        players_df = players_df[relevant_features].copy()
        
        # Convert cost to actual value (given in tenths)
        players_df['now_cost'] = players_df['now_cost'] / 10
        
        # Add position names
        position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        players_df['position'] = players_df['element_type'].map(position_map)
        
        # Add fixture difficulty
        players_df = players_df.merge(fixture_difficulty, on='team', how='left')
        
        # Add team name
        players_df = players_df.merge(
            teams_df[['id', 'name']], 
            left_on='team', 
            right_on='id', 
            suffixes=('', '_team')
        )
        players_df = players_df.rename(columns={'name': 'team_name'})
        
        return players_df
    
    @staticmethod
    def save_data(df: pd.DataFrame, filename: str = 'fpl_data.csv'):
        """Save processed data to CSV file."""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        df.to_csv(data_dir / filename, index=False)
        print(f"Data saved to {data_dir / filename}")

def fetch_and_process_data():
    """Main function to fetch and process FPL data."""
    fetcher = FPLDataFetcher()
    try:
        # Fetch both basic data and fixtures
        raw_data = fetcher.fetch_bootstrap_static()
        fixtures_data = fetcher.fetch_fixtures()
        
        # Calculate fixture difficulty
        fixture_difficulty = fetcher.calculate_fixture_difficulty(fixtures_data)
        
        # Process player data with fixture information
        processed_data = fetcher.process_player_data(raw_data, fixture_difficulty)
        fetcher.save_data(processed_data)
        return processed_data
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    fetch_and_process_data() 