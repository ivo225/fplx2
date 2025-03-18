from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY
import sys
from datetime import datetime, timedelta

def test_youtube_api():
    print(f"Testing YouTube API key: {YOUTUBE_API_KEY[:5]}...")
    try:
        # Initialize the YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Test 1: Try to get info about a channel
        print("\nTest 1: Checking channel info...")
        request = youtube.channels().list(
            part="snippet",
            id="UCNAf1k0yIjyGu3k9BwAg3lg"
        )
        response = request.execute()
        
        if response.get('items'):
            print("✓ Successfully retrieved channel information")
            print(f"Channel title: {response['items'][0]['snippet']['title']}")
        else:
            print("✗ API request successful but no channel data returned")
            return False

        # Test 2: Try to search for videos
        print("\nTest 2: Checking search functionality...")
        published_after = (datetime.utcnow() - timedelta(days=3)).isoformat() + 'Z'
        request = youtube.search().list(
            part='snippet',
            channelId="UCNAf1k0yIjyGu3k9BwAg3lg",
            maxResults=1,
            order='date',
            publishedAfter=published_after,
            type='video'
        )
        response = request.execute()
        
        if response.get('items'):
            print("✓ Successfully retrieved search results")
            video = response['items'][0]
            print(f"Latest video: {video['snippet']['title']}")
            return True
        else:
            print("✗ API request successful but no search results returned")
            return False
            
    except Exception as e:
        print(f"✗ Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_youtube_api()
    sys.exit(0 if success else 1) 