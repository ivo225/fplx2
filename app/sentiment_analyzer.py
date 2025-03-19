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
from historical_data import HistoricalDataManager
import re

class FPLSentimentAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the sentiment analyzer with YouTube API key."""
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.channels = {
            'Let\'s Talk FPL': 'UCxeOc7eFxq37yW_Nc-69deA',      # 444K subs
            'FPL Mate': 'UCweDAlFm2LnVcOqaFU4_AGA',             # 220K subs
            'FPLtips': 'UCVPb_jLxwaoYd-Dm7aSWQKQ',             # 219K subs
            'FPLHarry': 'UCcPWnCj5AKC19HaySZjb25g',            # 160K subs
            'FPL Raptor': 'UC54QLWzsMifTRjNQ02z5pCw',          # 147K subs
            'Fantasy Football Hub': 'UCcqEr3DfrRwtoF2a1yW8qgQ',  # 101K subs
            'Fantasy Football Scout': 'UCKxYKQ8pgJ7V8wwh4hLsSXQ', # 96K subs
            'FPL BlackBox': 'UCGJ8-xqhOLwyJNuPMsVoQWQ',         # 35K subs
            'Planet FPL': 'UC8043oOKTB4uP8Nq15Kz6bg',          # 19K subs
            'Above Average FPL': 'UCnaJiRMf5hju0TlaeGK5CDQ',    # 9K subs
            'FPL D-Unit': 'UCHNLdprRmMZLlE_Gx9OtRUw'           # 840 subs
        }
        self.channel_ids = {}  # Cache for resolved channel IDs
        self.cache_file = Path('data/sentiment_cache.json')
        self.quota_file = Path('data/youtube_quota.json')
        self.daily_quota_limit = 10000  # Default YouTube API quota limit
        self.quota_used = self._load_quota_usage()
        self.historical_data = HistoricalDataManager()
        
    def _load_quota_usage(self) -> int:
        """Load the current day's quota usage."""
        try:
            if not self.quota_file.exists():
                return 0

            with open(self.quota_file, 'r') as f:
                quota_data = json.load(f)
                
            # Check if quota data is from today
            last_reset = datetime.fromisoformat(quota_data.get('last_reset', '2000-01-01'))
            if last_reset.date() != datetime.now().date():
                return 0
                
            return quota_data.get('quota_used', 0)
        except Exception as e:
            print(f"\nError loading quota data: {str(e)}")
            return 0

    def _load_cache(self) -> Dict:
        """Load the sentiment cache file."""
        try:
            if not self.cache_file.exists():
                return {}
                
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"\nError loading sentiment cache: {str(e)}")
            return {}

    def _save_cache(self, cache_data: Dict) -> None:
        """Save data to the sentiment cache file."""
        try:
            # Ensure the data directory exists
            self.cache_file.parent.mkdir(exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"\nError saving to sentiment cache: {str(e)}")

    def _update_quota_usage(self, units: int):
        """Update the quota usage tracking."""
        try:
            self.quota_used += units
            
            # Ensure data directory exists
            self.quota_file.parent.mkdir(exist_ok=True)
            
            quota_data = {
                'last_reset': datetime.now().isoformat(),
                'quota_used': self.quota_used
            }
            
            with open(self.quota_file, 'w') as f:
                json.dump(quota_data, f, indent=2)
                
        except Exception as e:
            print(f"\nError updating quota data: {str(e)}")

    def _check_quota_available(self, required_units: int) -> bool:
        """Check if we have enough quota available."""
        return (self.quota_used + required_units) <= self.daily_quota_limit

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
        
    def _resolve_channel_id(self, channel_name: str, channel_username: str) -> str:
        """Resolve channel ID from username."""
        if channel_name in self.channel_ids:
            return self.channel_ids[channel_name]

        try:
            print(f"\nResolving channel ID for {channel_name} ({channel_username})...")
            
            # First try to get channel by username
            request = self.youtube.channels().list(
                part='id',
                forUsername=channel_username.replace('@', '')
            )
            response = request.execute()
            self._update_quota_usage(1)  # Channel lookup costs 1 unit

            if response.get('items'):
                channel_id = response['items'][0]['id']
                self.channel_ids[channel_name] = channel_id
                print(f"✅ Found channel ID via username: {channel_id}")
                return channel_id

            print("Channel not found by username, trying search...")
            # If username lookup fails, try search
            request = self.youtube.search().list(
                part='snippet',
                q=channel_username,
                type='channel',
                maxResults=1
            )
            response = request.execute()
            self._update_quota_usage(100)  # Search costs 100 units

            if response.get('items'):
                channel_id = response['items'][0]['snippet']['channelId']
                self.channel_ids[channel_name] = channel_id
                print(f"✅ Found channel ID via search: {channel_id}")
                return channel_id

            raise Exception(f"Could not find channel ID for {channel_username}")

        except Exception as e:
            print(f"⚠️ Error resolving channel ID for {channel_username}: {str(e)}")
            return None

    def get_recent_videos(self, days_back: int = 7, target_gameweek: int = None) -> List[Dict]:
        """Get recent FPL-related videos from monitored channels."""
        print("\n=== Starting Video Fetch ===")
        print(f"Looking back {days_back} days")
        print(f"Target Gameweek: {target_gameweek}")
        print(f"Monitoring {len(self.channels)} channels")
        
        videos = []
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + 'Z'
        
        # Build search terms for the specific gameweek
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
            print("Search terms:", gameweek_terms)
        
        # Track channels we couldn't process
        failed_channels = []
        quota_exceeded = False
        permission_denied_channels = []
        
        for channel_name, channel_id in self.channels.items():
            print(f"\n📺 Processing channel: {channel_name}")
            
            # Check quota
            if not self._check_quota_available(100):
                print("\n⚠️ YouTube API daily quota limit reached!")
                print(f"Processed {len(self.channels) - len(failed_channels) - len(permission_denied_channels)} channels")
                print(f"Remaining channels will be skipped: {', '.join(list(self.channels.keys())[len(videos):])}")
                quota_exceeded = True
                break
            
            retry_count = 0
            max_retries = 3
            base_delay = 2
            
            while retry_count < max_retries:
                try:
                    print(f"🔍 Searching for videos...")
                    request = self.youtube.search().list(
                        part='snippet',
                        channelId=channel_id,
                        maxResults=30,
                        order='date',
                        publishedAfter=published_after,
                        type='video'
                    )
                    response = request.execute()
                    self._update_quota_usage(100)
                    
                    if 'error' in response:
                        error_msg = response['error'].get('message', 'Unknown API error')
                        print(f"⚠️ API Error: {error_msg}")
                        if 'quotaExceeded' in error_msg:
                            quota_exceeded = True
                            break
                        raise Exception(error_msg)
                    
                    # Process videos
                    channel_videos = []
                    for item in response.get('items', []):
                        video_id = item['id']['videoId']
                        title = item['snippet']['title']
                        description = item['snippet']['description']
                        published_at = datetime.strptime(
                            item['snippet']['publishedAt'], 
                            '%Y-%m-%dT%H:%M:%SZ'
                        )
                        
                        # Check relevance
                        is_target_gameweek = False
                        if gameweek_terms and any(term.lower() in title.lower() for term in gameweek_terms):
                            is_target_gameweek = True
                            relevance_score = 10
                            print(f"🎯 Found gameweek-specific video: {title}")
                        
                        is_fpl_related = any(term.lower() in title.lower() or term.lower() in description.lower() 
                                           for term in ['fpl', 'fantasy', 'premier league', 'transfer', 'captain'])
                        
                        if is_target_gameweek or is_fpl_related:
                            # Get video comments if we have quota available
                            comments = []
                            if self._check_quota_available(1):
                                try:
                                    comments = self.get_video_comments(video_id)
                                    print(f"💬 Found {len(comments)} comments for video")
                                except Exception as e:
                                    print(f"⚠️ Error fetching comments: {str(e)}")
                            
                            video_data = {
                                'video_id': video_id,
                                'title': title,
                                'description': description,
                                'channel': channel_name,
                                'published_at': published_at,
                                'relevance': 10 if is_target_gameweek else 5,
                                'comments': comments
                            }
                            
                            # Recency boost
                            days_old = (datetime.utcnow() - published_at).days
                            if days_old <= 1:
                                video_data['relevance'] += 3
                                print("📅 Added recency boost (1 day old)")
                            elif days_old <= 2:
                                video_data['relevance'] += 2
                                print("📅 Added recency boost (2 days old)")
                            elif days_old <= 3:
                                video_data['relevance'] += 1
                                print("📅 Added recency boost (3 days old)")
                                
                            channel_videos.append(video_data)
                            print(f"✅ Added relevant video: {title}")
                    
                    videos.extend(channel_videos)
                    print(f"📊 Added {len(channel_videos)} videos from {channel_name}")
                    break
                    
                except HttpError as e:
                    print(f"⚠️ HTTP Error: {e.resp.status} {e.resp.reason}")
                    if e.resp.status == 403:
                        if "quotaExceeded" in str(e.content):
                            print("YouTube API quota exceeded")
                            quota_exceeded = True
                        else:
                            print("Permission denied")
                            permission_denied_channels.append(channel_name)
                        break
                    elif e.resp.status == 429:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = base_delay * (2 ** retry_count)
                            print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print("Max retries exceeded")
                            failed_channels.append(channel_name)
                            break
                    else:
                        print("Unhandled HTTP error")
                        failed_channels.append(channel_name)
                        break
                except Exception as e:
                    print(f"⚠️ Unexpected error: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = base_delay * (2 ** retry_count)
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded")
                        failed_channels.append(channel_name)
                        break
            
            if quota_exceeded:
                break
            
            retry_count = 0
            time.sleep(base_delay)
        
        # Sort and limit videos
        videos.sort(key=lambda x: (x['relevance'], x['published_at']), reverse=True)
        max_videos = 50
        if len(videos) > max_videos:
            videos = videos[:max_videos]
            
        # Print summary
        print("\n=== Video Fetch Summary ===")
        print(f"Total channels processed: {len(self.channels) - len(failed_channels) - len(permission_denied_channels)}")
        if permission_denied_channels:
            print(f"\n⚠️ Channels with permission issues ({len(permission_denied_channels)}):")
            for channel in permission_denied_channels:
                print(f"  • {channel}")
        if failed_channels:
            print(f"\n❌ Failed channels ({len(failed_channels)}):")
            for channel in failed_channels:
                print(f"  • {channel}")
        
        print(f"\nTotal videos found: {len(videos)}")
        print(f"YouTube API quota used: {self.quota_used}/{self.daily_quota_limit}")
        
        if quota_exceeded:
            print("\n⚠️ Some channels were skipped due to YouTube API quota limits")
            
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
            comments = video['comments']
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

    def analyze_sentiment(self, target_gameweek: int = None) -> Dict[str, float]:
        """Analyze sentiment for FPL-related content."""
        print("\n=== Starting Sentiment Analysis ===")
        print(f"Target Gameweek: {target_gameweek}")
        
        # Check cache first
        cache_key = f'gw{target_gameweek}' if target_gameweek else 'latest'
        cached_data = self._load_cache()
        
        if cache_key in cached_data:
            cached_result = cached_data[cache_key]
            # Check if cache is from today
            cache_date = datetime.fromisoformat(cached_result['timestamp'])
            if cache_date.date() == datetime.utcnow().date():
                print(f"\nUsing cached sentiment scores for {cache_key}")
                return cached_result['sentiment_scores']
        
        print("\n🔍 Fetching fresh video data...")
        print("YouTube API Key status:", "Available" if self.youtube else "Missing")
        print("Current API quota used:", self.quota_used)
        print("Daily quota limit:", self.daily_quota_limit)
        
        # Fetch and analyze new videos
        videos = self.get_recent_videos(days_back=7, target_gameweek=target_gameweek)
        if not videos:
            print("\n⚠️ No videos found - using neutral sentiment scores")
            return {}
            
        print(f"\n✅ Successfully fetched {len(videos)} videos")
        
        # Calculate sentiment scores
        sentiment_scores = {}
        print(f"\n📊 Analyzing {len(videos)} videos for sentiment...")
        
        for video in videos:
            print(f"\n🎥 Processing video: {video['title']}")
            
            # Analyze video title and description
            title_sentiment = self.analyze_text_sentiment(video['title'])[0]
            desc_sentiment = self.analyze_text_sentiment(video['description'])[0]
            
            # Get and analyze comments
            comments = video.get('comments', [])
            comment_sentiments = [self.analyze_text_sentiment(comment)[0] for comment in comments]
            avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0
            
            # Extract player names from title and description
            content = f"{video['title']} {video['description']}".lower()
            
            # Common FPL player name patterns
            patterns = [
                r'[A-Z][a-z]+ [A-Z][a-z]+',  # Full names (e.g., "Mohamed Salah")
                r'[A-Z][a-z]+\.',  # Abbreviated names (e.g., "Salah.")
                r'[A-Z]\. [A-Z][a-z]+',  # Initial + surname (e.g., "M. Salah")
                r'[A-Z][a-zA-Z]+',  # Single names (e.g., "Salah", "TAA")
                r'[A-Z][a-z]+',  # Single names with proper capitalization
                r'van [A-Z][a-z]+',  # Dutch names (e.g., "van Dijk")
                r'de [A-Z][a-z]+',  # Spanish/Portuguese names (e.g., "de Bruyne")
                r'[A-Z][a-z]+-[A-Z][a-z]+',  # Hyphenated names (e.g., "Saint-Maximin")
            ]
            
            # Common FPL player nicknames and abbreviations
            nicknames = {
                'taa': 'Alexander-Arnold',
                'kdb': 'De Bruyne',
                'vvd': 'van Dijk',
                'asm': 'Saint-Maximin',
                'dcl': 'Calvert-Lewin',
                'trossard': 'Trossard',
                'martinelli': 'Martinelli',
                'saka': 'Saka',
                'haaland': 'Haaland',
                'salah': 'Salah',
                'kane': 'Kane',
                'son': 'Son',
                'rashford': 'Rashford',
                'bruno': 'Fernandes',
                'foden': 'Foden',
                'grealish': 'Grealish',
                'watkins': 'Watkins',
                'bowen': 'Bowen',
                'gordon': 'Gordon',
                'isak': 'Isak',
                'wilson': 'Wilson',
                'nunez': 'Núñez',
                'diaz': 'Díaz',
                'jota': 'Jota',
                'darwin': 'Núñez',
                'toney': 'Toney',
                'mbeumo': 'Mbeumo',
                'palmer': 'Palmer',
                'jackson': 'Jackson',
                'sterling': 'Sterling',
                'mudryk': 'Mudryk',
                'ollie': 'Watkins',
                'mitro': 'Mitrović',
                'mitroma': 'Mitoma',
                'macca': 'Mac Allister',
                'kulu': 'Kulusevski',
                'madders': 'Maddison',
            }
            
            # Extract potential player names from title and description
            player_names = set()
            
            # First check for nicknames
            content_words = content.split()
            for word in content_words:
                word = word.lower().strip('.,!?')
                if word in nicknames:
                    player_names.add(nicknames[word])
                    print(f"🎯 Found player via nickname: {word} -> {nicknames[word]}")
            
            # Then check regex patterns
            for pattern in patterns:
                matches = re.findall(pattern, video['title'] + " " + video['description'])
                for match in matches:
                    player_names.add(match)
                    print(f"🎯 Found player via pattern: {match}")
            
            # Process each potential player name
            for player_name in player_names:
                player_key = player_name.lower()
                if player_key not in sentiment_scores:
                    sentiment_scores[player_key] = {
                        'mentions': 0,
                        'sentiment': 0.0,
                        'display_name': player_name
                    }
                
                # Weight title mentions more heavily
                if player_name.lower() in video['title'].lower():
                    sentiment_scores[player_key]['mentions'] += 2
                    sentiment_scores[player_key]['sentiment'] += title_sentiment * 2
                    print(f"🔥 Found {player_name} in video title with sentiment {title_sentiment:.2f}")
                
                # Add description sentiment
                if player_name.lower() in video['description'].lower():
                    sentiment_scores[player_key]['mentions'] += 1
                    sentiment_scores[player_key]['sentiment'] += desc_sentiment
                    print(f"📝 Found {player_name} in description with sentiment {desc_sentiment:.2f}")
                
                # Add comment sentiments
                for comment in comments:
                    if player_name.lower() in comment.lower():
                        sentiment_scores[player_key]['mentions'] += 0.5
                        sentiment_scores[player_key]['sentiment'] += avg_comment_sentiment * 0.5
                        print(f"💬 Found {player_name} in comments with sentiment {avg_comment_sentiment:.2f}")
        
        # Calculate final normalized sentiment scores
        final_scores = {}
        print("\n🏆 Final Sentiment Scores:")
        for player_key, data in sentiment_scores.items():
            if data['mentions'] > 0:
                avg_sentiment = data['sentiment'] / data['mentions']
                # Normalize to range [0.7, 1.3]
                normalized_sentiment = 1.0 + (avg_sentiment * 0.3)
                final_score = max(0.7, min(1.3, normalized_sentiment))
                final_scores[data['display_name']] = final_score
                
                # Print with emoji indicators
                sentiment_emoji = "🔥" if final_score > 1.1 else "📈" if final_score > 1.0 else "📉" if final_score < 0.9 else "➖"
                print(f"{sentiment_emoji} {data['display_name']}: {final_score:.2f} (from {data['mentions']} mentions)")
        
        # Prepare data to cache
        cache_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'sentiment_scores': final_scores
        }
        
        # Update cache
        cached_data[cache_key] = cache_data
        self._save_cache(cached_data)
        
        print("\n=== Sentiment Analysis Complete ===")
        return final_scores
    
    def _extract_player_names(self, content: str) -> List[str]:
        """Extract player names from content using common FPL player references."""
        # This is a simple implementation - could be enhanced with more sophisticated NLP
        player_names = []
        
        # Common FPL player name patterns
        patterns = [
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # Full names
            r'[A-Z][a-z]+\.',  # Abbreviated names
            r'[A-Z]\. [A-Z][a-z]+',  # Initial + surname
        ]
        
        # Extract potential player names
        for pattern in patterns:
            matches = re.findall(pattern, content)
            player_names.extend(matches)
        
        return list(set(player_names))  # Remove duplicates

    def _get_current_gameweek(self) -> int:
        """Get the current gameweek number."""
        # This is a placeholder - implement actual gameweek detection
        return 30  # Temporary hardcoded value

def create_analyzer(api_key: str) -> FPLSentimentAnalyzer:
    """Create and return a sentiment analyzer instance."""
    return FPLSentimentAnalyzer(api_key) 