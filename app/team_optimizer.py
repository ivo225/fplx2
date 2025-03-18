import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pulp import *
from sentiment_analyzer import create_analyzer
from config import YOUTUBE_API_KEY, SENTIMENT_LOOKBACK_DAYS, MAX_COMMENTS_PER_VIDEO

@dataclass
class TeamConstraints:
    TOTAL_BUDGET: float = 100.0
    MAX_PER_TEAM: int = 3
    SQUAD_SIZE: int = 15
    MIN_HIGH_SENTIMENT_PLAYERS: int = 4  # Minimum number of high-sentiment players in starting XI
    HIGH_SENTIMENT_THRESHOLD: float = 1.05  # Threshold for considering a player as having high sentiment
    # Position-specific sentiment thresholds
    POSITION_SENTIMENT_THRESHOLDS = {
        'FWD': 1.03,  # Strikers need less sentiment as they're more fixture-dependent
        'MID': 1.04,  # Midfielders slightly higher
        'DEF': 1.03,  # Lowered from 1.05 to match forwards since defensive stability is key
        'GKP': 1.05   # Keepers are long-term picks
    }
    FORMATION = {
        'GKP': 2,
        'DEF': 5,  # Increased to 5 defenders (3 starting + 2 bench)
        'MID': 5,
        'FWD': 3   # Total forwards in squad (all starting in a 3-4-3)
    }
    STARTING_FORMATION = {
        'GKP': 1,
        'DEF': 3,  # 3-4-3 formation
        'MID': 4,
        'FWD': 3
    }

class TeamOptimizer:
    def __init__(self, players_df: pd.DataFrame, predicted_points: np.ndarray, target_gameweek: int = None, force_refresh_sentiment: bool = False):
        """Initialize the team optimizer.
        
        Args:
            players_df: DataFrame containing player information
            predicted_points: Array of predicted points for each player
            target_gameweek: The gameweek number to optimize for (used for sentiment analysis)
            force_refresh_sentiment: If True, forces recalculation of sentiment scores ignoring cache
        """
        self.constraints = TeamConstraints()
        self.players_df = players_df.copy()
        self.players_df['predicted_points'] = predicted_points
        
        # Get sentiment scores
        try:
            analyzer = create_analyzer(YOUTUBE_API_KEY)
            sentiment_scores = analyzer.calculate_player_sentiment(
                self.players_df,
                target_gameweek=target_gameweek,
                force_refresh=force_refresh_sentiment
            )
            self.players_df['sentiment_score'] = self.players_df['web_name'].map(sentiment_scores)
            print("\nSentiment Analysis Results:")
            high_sentiment = self.players_df[self.players_df['sentiment_score'] > self.constraints.HIGH_SENTIMENT_THRESHOLD]
            if not high_sentiment.empty:
                print("\nPlayers with High Sentiment (>1.05):")
                for _, player in high_sentiment.sort_values('sentiment_score', ascending=False).iterrows():
                    print(f"{player['web_name']} ({player['team_name']}) - Sentiment: {player['sentiment_score']:.2f}")
            else:
                print("\nNo players found with high sentiment scores (>1.05)")
        except ValueError as e:
            print(f"\nWarning: Sentiment analysis failed: {str(e)}")
            print("Using neutral sentiment scores for all players.")
            self.players_df['sentiment_score'] = 1.0
        except Exception as e:
            print(f"\nError in sentiment analysis: {str(e)}")
            print("Please check your YouTube API key and internet connection.")
            print("Using neutral sentiment scores for all players.")
            self.players_df['sentiment_score'] = 1.0
        
        # Filter out injured players and those not available
        self.players_df['is_available'] = ~(
            self.players_df['chance_of_playing_next_round'].isin([0, 25, 50]) |  # Injured or doubtful
            self.players_df['status'].str.contains('i|d', case=False, na=False)  # Status contains 'injured' or 'doubtful'
        )
        
        print("\nExcluded players due to injuries/availability:")
        unavailable = self.players_df[~self.players_df['is_available']]
        for _, player in unavailable.iterrows():
            print(f"{player['web_name']} ({player['team_name']}) - {player['status']}")
        
        # Keep only available players
        self.players_df = self.players_df[self.players_df['is_available']].copy()
        
        # Convert form to float and fill NaN with 0
        self.players_df['form'] = self.players_df['form'].astype(float).fillna(0)
        
        print("\nPlayers in excellent form (form > 5.0):")
        good_form = self.players_df[self.players_df['form'] > 5.0]
        for _, player in good_form.iterrows():
            print(f"{player['web_name']} ({player['team_name']}) - Form: {player['form']}")
    
    def optimize_team(self) -> pd.DataFrame:
        """Build optimal team using PuLP linear programming."""
        if len(self.players_df) == 0:
            raise ValueError("No available players found after filtering injuries!")
            
        # Create optimization problem
        prob = LpProblem("FPL_Team_Selection", LpMaximize)
        
        # Create binary variables for each player (1 if selected, 0 if not)
        player_vars = LpVariable.dicts("player",
                                     ((i, status) for i in self.players_df.index 
                                      for status in ['starting', 'bench']),
                                     cat='Binary')
        
        # Enhanced objective function with stronger sentiment influence and proper error handling
        prob += lpSum([
            self.players_df.loc[i, 'predicted_points'] * 
            (1.5 if self.players_df.loc[i, 'fixture_difficulty'] <= 2.5 else 1.0) * 
            (1.0 + max(0, float(self.players_df.loc[i, 'form']) - 4) * 0.1) *  
            (self.players_df.loc[i, 'sentiment_score'] ** 2.0) *  # Increased sentiment power for starting XI
            (1.2 if (
                not pd.isna(self.players_df.loc[i, 'position']) and
                self.players_df.loc[i, 'position'] in self.constraints.POSITION_SENTIMENT_THRESHOLDS and
                self.players_df.loc[i, 'sentiment_score'] > 
                self.constraints.POSITION_SENTIMENT_THRESHOLDS[self.players_df.loc[i, 'position']]
            ) else 1.0) *  # Position-specific bonus with error handling
            player_vars[i, 'starting'] +
            0.1 * self.players_df.loc[i, 'predicted_points'] * 
            (1.2 if self.players_df.loc[i, 'fixture_difficulty'] <= 2.5 else 1.0) * 
            (1.0 + max(0, float(self.players_df.loc[i, 'form']) - 4) * 0.05) *
            self.players_df.loc[i, 'sentiment_score'] *  # Normal sentiment impact for bench
            player_vars[i, 'bench']
            for i in self.players_df.index
        ])
        
        # Constraint 1: Total budget
        prob += lpSum([self.players_df.loc[i, 'now_cost'] * 
                      (player_vars[i, 'starting'] + player_vars[i, 'bench'])
                      for i in self.players_df.index]) <= self.constraints.TOTAL_BUDGET
        
        # Constraint 2: Squad size constraints for each position
        for position in self.constraints.FORMATION:
            position_indices = self.players_df[self.players_df['position'] == position].index
            
            # Total players in position
            prob += lpSum([player_vars[i, 'starting'] + player_vars[i, 'bench']
                         for i in position_indices]) == self.constraints.FORMATION[position]
            
            # Starting players in position
            prob += lpSum([player_vars[i, 'starting']
                         for i in position_indices]) == self.constraints.STARTING_FORMATION[position]
        
        # Constraint 3: Maximum players per team
        for team in self.players_df['team'].unique():
            team_indices = self.players_df[self.players_df['team'] == team].index
            prob += lpSum([player_vars[i, 'starting'] + player_vars[i, 'bench']
                         for i in team_indices]) <= self.constraints.MAX_PER_TEAM
        
        # Constraint 4: Each player can only be selected once
        for i in self.players_df.index:
            prob += player_vars[i, 'starting'] + player_vars[i, 'bench'] <= 1
        
        # Constraint 5: Average fixture difficulty for starting XI should be reasonable
        prob += (lpSum([self.players_df.loc[i, 'fixture_difficulty'] * player_vars[i, 'starting']
                      for i in self.players_df.index]) / 11) <= 2.8  # Maximum average difficulty
        
        # Constraint 6: Minimum number of high-sentiment players in starting XI
        high_sentiment_indices = self.players_df[
            self.players_df['sentiment_score'] > self.constraints.HIGH_SENTIMENT_THRESHOLD
        ].index
        if len(high_sentiment_indices) >= self.constraints.MIN_HIGH_SENTIMENT_PLAYERS:
            prob += lpSum([player_vars[i, 'starting']
                         for i in high_sentiment_indices]) >= self.constraints.MIN_HIGH_SENTIMENT_PLAYERS
        
        # Solve the problem
        print("\nOptimizing team selection...")
        prob.solve()
        
        # Extract results
        selected_players = []
        for i in self.players_df.index:
            if value(player_vars[i, 'starting']) == 1:
                player = self.players_df.loc[i].to_dict()
                player['status'] = 'Starting'
                selected_players.append(player)
            elif value(player_vars[i, 'bench']) == 1:
                player = self.players_df.loc[i].to_dict()
                player['status'] = 'Bench'
                selected_players.append(player)
        
        # Convert to DataFrame
        optimal_df = pd.DataFrame(selected_players)
        
        # Calculate team statistics
        total_predicted_points = optimal_df['predicted_points'].sum()
        total_cost = optimal_df['now_cost'].sum()
        avg_difficulty = optimal_df['fixture_difficulty'].mean()
        avg_sentiment = optimal_df[optimal_df['status'] == 'Starting']['sentiment_score'].mean()
        high_sentiment_count = len(optimal_df[
            (optimal_df['status'] == 'Starting') & 
            (optimal_df['sentiment_score'] > self.constraints.HIGH_SENTIMENT_THRESHOLD)
        ])
        
        print(f"\nTeam Statistics:")
        print(f"Total Predicted Points: {total_predicted_points:.2f}")
        print(f"Total Cost: £{total_cost:.1f}M")
        print(f"Remaining Budget: £{(self.constraints.TOTAL_BUDGET - total_cost):.1f}M")
        print(f"Average Fixture Difficulty: {avg_difficulty:.2f} (1=Easy, 5=Hard)")
        print(f"Average Starting XI Sentiment: {avg_sentiment:.2f}")
        print(f"High Sentiment Players in Starting XI: {high_sentiment_count}")
        
        # Sort by position and status
        position_order = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        optimal_df['position_order'] = optimal_df['position'].map(position_order)
        
        # Assign bench numbers
        bench_players = optimal_df[optimal_df['status'] == 'Bench']
        # Prioritize positions for bench slots
        bench_status = {
            'GKP': 'GK',  # Reserve goalkeeper
            'DEF': 'B1',  # First outfield sub (defense)
            'MID': 'B2',  # Second outfield sub (midfield)
        }
        
        # Assign bench numbers based on available positions
        bench_positions = bench_players['position'].unique()
        
        # Always assign GK to the goalkeeper
        gk_bench = bench_players[bench_players['position'] == 'GKP']
        if not gk_bench.empty:
            optimal_df.loc[gk_bench.index[0], 'status'] = 'GK'
        
        # Assign B1, B2, B3 to outfield players based on position priority: DEF, MID, FWD
        # (in a 3-4-3 formation with 15 players, we'll have 2 DEF and 1 MID on bench)
        bench_outfield = bench_players[bench_players['position'] != 'GKP']
        
        # Sort by position order (DEF first, then MID, then FWD if any)
        bench_outfield = bench_outfield.sort_values('position_order')
        
        # Assign bench numbers in order
        bench_numbers = ['B1', 'B2', 'B3']
        for i, (idx, player) in enumerate(bench_outfield.iterrows()):
            if i < len(bench_numbers):
                optimal_df.loc[idx, 'status'] = bench_numbers[i]
        
        # Sort the team
        optimal_df = optimal_df.sort_values(
            ['position_order', 'status', 'predicted_points'],
            ascending=[True, True, False]
        )
        
        # Format fixture difficulty for display
        optimal_df['fixtures'] = optimal_df.apply(
            lambda x: f"{x['team_name']} (Difficulty: {x['fixture_difficulty']:.1f})",
            axis=1
        )
        
        # Calculate points per cost
        optimal_df['points_per_cost'] = optimal_df['predicted_points'] / optimal_df['now_cost']
        
        # Add sentiment indicator to display
        optimal_df['sentiment_indicator'] = optimal_df['sentiment_score'].apply(
            lambda x: f"🔥 {x:.2f}" if x > self.constraints.HIGH_SENTIMENT_THRESHOLD else f"{x:.2f}"
        )
        
        return optimal_df[['web_name', 'position', 'status', 'fixtures', 'now_cost',
                          'predicted_points', 'points_per_cost', 'sentiment_score', 'sentiment_indicator']] 