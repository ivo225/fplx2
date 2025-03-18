from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import os
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

class FPLSentimentAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the sentiment analyzer with YouTube API key."""
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.channels = {
            'Premier League': 'UCNAf1k0yIjyGu3k9BwAg3lg',
            'Fantasy Premier League': 'UCt0ybDqVhTnKTwzZ8XRqudg',
            'Let\'s Talk FPL': 'UCmUZqU0qz7RYG7wZZqUtX7Q',
            'FPL Mate': 'UCB3c6zQqYcBRLHBcpWYWDFA',
            'FPL Raptor': 'UCPe5ahWbyYFP5t2YyKMZi3w'
        }
        self.cache_file = Path('data/sentiment_cache.json')
        
    def _load_cached_sentiment(self, cache_key: str = "default") -> Tuple[Dict[str, float], bool]:
        """Load cached sentiment scores if they exist and are from today.
        
        Args:
            cache_key: Key to identify the specific cache (e.g., "gw30" for gameweek 30)
        """
        try:
            if not self.cache_file.exists():
                return {}, False
                
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if we have a cache for this specific key
            if cache_key not in cache_data:
                print(f"\nNo cached sentiment scores found for {cache_key}")
                return {}, False
                
            cache = cache_data[cache_key]
            
            # Check if cache is from today
            cache_date = datetime.fromisoformat(cache['timestamp'])
            if cache_date.date() == datetime.now().date():
                print(f"\nUsing cached sentiment scores for {cache_key} from today")
                return cache['sentiment_scores'], True
            else:
                print(f"\nCached sentiment scores for {cache_key} are outdated")
                return {}, False
                
        except Exception as e:
            print(f"\nError loading cached sentiment: {str(e)}")
            return {}, False
            
    def _save_sentiment_cache(self, sentiment_scores: Dict[str, float], cache_key: str = "default"):
        """Save sentiment scores to cache file.
        
        Args:
            sentiment_scores: Dictionary of player sentiment scores
            cache_key: Key to identify the specific cache (e.g., "gw30" for gameweek 30)
        """
        try:
            # Create data directory if it doesn't exist
            self.cache_file.parent.mkdir(exist_ok=True)
            
            # Load existing cache data if it exists
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    try:
                        cache_data = json.load(f)
                    except json.JSONDecodeError:
                        # File exists but is not valid JSON, start fresh
                        cache_data = {}
            
            # Update the specific cache entry
            cache_data[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'sentiment_scores': sentiment_scores
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"\nSaved sentiment scores for {cache_key} to cache")
            
        except Exception as e:
            print(f"\nError saving sentiment cache: {str(e)}")
        
    def get_recent_videos(self, days_back: int = 7, target_gameweek: int = None) -> List[Dict]:
        """Get recent FPL-related videos from monitored channels.
        
        Args:
            days_back: Number of days to look back for videos
            target_gameweek: Specific gameweek number to focus on (e.g., GW30)
        """
        videos = []
        # Set to fetch videos published in the last week to ensure we don't miss relevant content
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + 'Z'
        
        # Build search terms for the specific gameweek if provided
        gameweek_terms = []
        if target_gameweek is not None:
            gameweek_terms = [
                f"GW{target_gameweek}",
                f"GW {target_gameweek}",
                f"Gameweek {target_gameweek}",
                f"Gameweek{target_gameweek}",
                f"Game Week {target_gameweek}",
                f"Game-Week {target_gameweek}"
            ]
            print(f"\nSearching for videos about Gameweek {target_gameweek}")
        
        for channel_name, channel_id in self.channels.items():
            try:
                print(f"\nFetching videos from {channel_name}...")
                # Search for videos from this channel
                request = self.youtube.search().list(
                    part='snippet',
                    channelId=channel_id,
                    maxResults=30,  # Increased to get more potential matches
                    order='date',
                    publishedAfter=published_after,
                    type='video'
                )
                response = request.execute()
                
                # Add videos to our list with proper date handling
                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    published_at = datetime.strptime(
                        item['snippet']['publishedAt'], 
                        '%Y-%m-%dT%H:%M:%SZ'
                    )
                    
                    # Check if this is a gameweek-specific video when target_gameweek is provided
                    is_target_gameweek = False
                    if gameweek_terms and any(term.lower() in title.lower() for term in gameweek_terms):
                        is_target_gameweek = True
                        relevance_score = 10  # High priority for gameweek-specific videos
                    
                    # Basic FPL relevance check
                    is_fpl_related = any(term.lower() in title.lower() or term.lower() in description.lower() 
                                         for term in ['fpl', 'fantasy', 'premier league', 'transfer', 'captain'])
                    
                    if is_target_gameweek or is_fpl_related:
                        # Add the video with a relevance score
                        video_data = {
                            'video_id': video_id,
                            'title': title,
                            'description': description,
                            'channel': channel_name,
                            'published_at': published_at,
                            'relevance': 10 if is_target_gameweek else 5  # Prioritize gameweek-specific videos
                        }
                        
                        # Additional boost for recency
                        days_old = (datetime.utcnow() - published_at).days
                        if days_old <= 1:  # Less than 1 day old
                            video_data['relevance'] += 3
                        elif days_old <= 2:  # Less than 2 days old
                            video_data['relevance'] += 2
                        elif days_old <= 3:  # Less than 3 days old
                            video_data['relevance'] += 1
                            
                        videos.append(video_data)
                        print(f"Found video ({published_at.strftime('%Y-%m-%d %H:%M')}): {title}" + 
                              (" [GAMEWEEK MATCH]" if is_target_gameweek else ""))
                
                # Add an exponential backoff delay between requests
                time.sleep(1)
                
            except HttpError as e:
                print(f"HTTP Error fetching videos from {channel_name}: {e.resp.status} {e.resp.reason}")
                if e.resp.status == 403:
                    print("Quota exceeded. Please wait or use a different API key.")
                    return videos  # Return what we have so far
                elif e.resp.status == 429:
                    print("Rate limit exceeded. Implementing exponential backoff...")
                    time.sleep(10)  # Longer wait for rate limits
            except Exception as e:
                print(f"Error fetching videos from {channel_name}: {str(e)}")
                
        # Sort videos by relevance (primary) and published date (secondary)
        videos.sort(key=lambda x: (x['relevance'], x['published_at']), reverse=True)
        
        # Take top videos, prioritizing gameweek-specific and recent ones
        max_videos = 50  # Reasonable limit to avoid processing too many videos
        if len(videos) > max_videos:
            videos = videos[:max_videos]
            
        # Print video dates as a summary
        video_dates = {}
        gameweek_specific_count = sum(1 for v in videos if v['relevance'] >= 10)
        
        for video in videos:
            date_str = video['published_at'].strftime('%Y-%m-%d')
            video_dates[date_str] = video_dates.get(date_str, 0) + 1
        
        print(f"\nFound {len(videos)} relevant videos ({gameweek_specific_count} gameweek-specific)")
        print("\nVideo date distribution:")
        for date, count in sorted(video_dates.items()):
            print(f"  {date}: {count} videos")
            
        return videos
    
    def get_video_comments(self, video_id: str, max_comments: int = 100) -> List[str]:
        """Get comments from a specific video."""
        comments = []
        try:
            print(f"\nFetching comments for video {video_id}...")
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_comments,
                textFormat='plainText',
                order='relevance'  # Get most relevant comments first
            )
            response = request.execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                text = comment['textDisplay']
                likes = comment.get('likeCount', 0)
                
                # Only include comments that are likely to be meaningful
                if len(text.split()) >= 3:  # Skip very short comments
                    comments.append({
                        'text': text,
                        'likes': likes,
                        'published_at': comment['publishedAt']
                    })
            
            # Sort comments by likes to prioritize community-validated opinions
            comments.sort(key=lambda x: x['likes'], reverse=True)
            
            # Extract just the text for the final list
            comments = [c['text'] for c in comments]
            
            print(f"Found {len(comments)} meaningful comments")
            
            # Add a small delay between requests
            time.sleep(1)
            
        except HttpError as e:
            print(f"HTTP Error fetching comments: {e.resp.status} {e.resp.reason}")
            if e.resp.status == 403:
                print("Quota exceeded. Please wait or use a different API key.")
            elif e.resp.status == 429:
                print("Rate limit exceeded. Implementing exponential backoff...")
                time.sleep(10)
        except Exception as e:
            print(f"Error fetching comments for video {video_id}: {str(e)}")
            
        return comments
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of a piece of text using TextBlob."""
        try:
            analysis = TextBlob(text)
            # Returns (polarity, subjectivity)
            # Polarity: -1 (negative) to 1 (positive)
            # Subjectivity: 0 (objective) to 1 (subjective)
            return analysis.sentiment.polarity, analysis.sentiment.subjectivity
        except Exception as e:
            print(f"Error analyzing text sentiment: {str(e)}")
            return 0.0, 0.0
    
    def calculate_player_sentiment(self, players_df: pd.DataFrame, target_gameweek: int = None, force_refresh: bool = False) -> Dict[str, float]:
        """Calculate sentiment scores for each player based on recent videos and comments.
        
        Args:
            players_df: DataFrame containing player information
            target_gameweek: The specific gameweek number to focus on (e.g., 30 for GW30)
            force_refresh: If True, bypasses cache and fetches fresh sentiment scores
        """
        # Try to load cached sentiment first (unless force_refresh is True)
        cache_key = f"gw{target_gameweek}" if target_gameweek else "default"
        
        if not force_refresh:
            cached_scores, is_valid = self._load_cached_sentiment(cache_key=cache_key)
            if is_valid:
                return cached_scores
        elif self.cache_file.exists():
            print(f"\nForce refresh requested - ignoring existing cache for {cache_key}")
            
        print("\nCalculating fresh player sentiment scores...")
        player_mentions = {player: {'mentions': 0, 'sentiment': 0.0} 
                         for player in players_df['web_name']}
        
        # Get recent videos
        videos = self.get_recent_videos(target_gameweek=target_gameweek)
        if not videos:
            raise ValueError(f"No recent FPL-related videos found{' for gameweek ' + str(target_gameweek) if target_gameweek else ''}. Please check the YouTube API key and video filtering.")
            
        print(f"\nAnalyzing {len(videos)} videos for player mentions...")
        
        for video in videos:
            # Analyze video title and description
            title_sentiment = self.analyze_text_sentiment(video['title'])[0]
            desc_sentiment = self.analyze_text_sentiment(video['description'])[0]
            
            # Get and analyze comments
            comments = self.get_video_comments(video['video_id'])
            comment_sentiments = [self.analyze_text_sentiment(comment)[0] for comment in comments]
            avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0
            
            # Check for player mentions
            for player in player_mentions:
                # Check title
                if player.lower() in video['title'].lower():
                    player_mentions[player]['mentions'] += 2  # Weight title mentions more
                    player_mentions[player]['sentiment'] += title_sentiment * 2
                    print(f"Found {player} in video title with sentiment {title_sentiment:.2f}")
                
                # Check description
                if player.lower() in video['description'].lower():
                    player_mentions[player]['mentions'] += 1
                    player_mentions[player]['sentiment'] += desc_sentiment
                    print(f"Found {player} in video description with sentiment {desc_sentiment:.2f}")
                
                # Check comments
                for comment in comments:
                    if player.lower() in comment.lower():
                        player_mentions[player]['mentions'] += 0.5  # Weight comments less
                        player_mentions[player]['sentiment'] += avg_comment_sentiment * 0.5
                        print(f"Found {player} in comments with average sentiment {avg_comment_sentiment:.2f}")
        
        # Calculate final sentiment scores
        print("\nCalculating final sentiment scores...")
        sentiment_scores = {}
        mentioned_players = 0
        for player, data in player_mentions.items():
            if data['mentions'] > 0:
                # Normalize sentiment to range [0.7, 1.3]
                avg_sentiment = data['sentiment'] / data['mentions']
                normalized_sentiment = 1.0 + (avg_sentiment * 0.3)  # Scale to [0.7, 1.3]
                sentiment_scores[player] = max(0.7, min(1.3, normalized_sentiment))
                print(f"{player}: {sentiment_scores[player]:.2f} (from {data['mentions']} mentions)")
                mentioned_players += 1
            else:
                sentiment_scores[player] = 1.0  # Neutral sentiment for unmentioned players
        
        if mentioned_players == 0:
            raise ValueError("No player mentions found in any videos. Please check the video content and player name matching.")
            
        print(f"\nFound sentiment scores for {mentioned_players} players")
        
        # Cache the results
        self._save_sentiment_cache(sentiment_scores, cache_key=cache_key)
        
        return sentiment_scores

def create_analyzer(api_key: str) -> FPLSentimentAnalyzer:
    """Create and return a sentiment analyzer instance."""
    return FPLSentimentAnalyzer(api_key) 