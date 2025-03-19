from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY
import sys
from datetime import datetime, timedelta

def test_youtube_api():
    channels = {
        'Let\'s Talk FPL': 'Let\'s Talk FPL',
        'FPL Mate': 'FPL Mate',
        'FPL Raptor': 'FPL Raptor',
        'FPL D-Unit': 'FPL D-Unit',
        'FPL BlackBox': 'FPL BlackBox',
        'Fantasy Football Hub': 'Fantasy Football Hub',
        'Above Average FPL': 'Above Average FPL',
        'FPLHarry': 'FPLHarry',
        'Planet FPL': 'Planet FPL',
        'FPLtips': 'FPLtips',
        'Fantasy Football Scout': 'Fantasy Football Scout'
    }
    
    print(f"Testing YouTube API key: {YOUTUBE_API_KEY[:5]}...")
    try:
        # Initialize the YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        channel_ids = {}
        
        for channel_name, search_term in channels.items():
            print(f"\nSearching for channel: {channel_name}")
            
            # Try to find channel
            request = youtube.search().list(
                part="snippet",
                q=search_term,
                type="channel",
                maxResults=5
            )
            response = request.execute()
            
            if response.get('items'):
                # Print all found channels to help verify
                print(f"Found {len(response['items'])} potential matches:")
                for idx, item in enumerate(response['items'], 1):
                    channel = item['snippet']
                    channel_id = item['id']['channelId']
                    print(f"{idx}. Title: {channel['title']}")
                    print(f"   Channel ID: {channel_id}")
                    print(f"   Description: {channel['description'][:100]}...")
                    
                    # Get subscriber count
                    channel_request = youtube.channels().list(
                        part="statistics",
                        id=channel_id
                    )
                    channel_response = channel_request.execute()
                    if channel_response.get('items'):
                        subs = int(channel_response['items'][0]['statistics']['subscriberCount'])
                        print(f"   Subscribers: {subs:,}")
                    print()
                
                # Store the first result's channel ID
                channel_ids[channel_name] = response['items'][0]['id']['channelId']
            else:
                print("❌ No channels found")
                
        # Print final results in a format ready for the sentiment analyzer
        print("\nChannel IDs for sentiment analyzer:")
        print("self.channels = {")
        for name, channel_id in channel_ids.items():
            print(f"    '{name}': '{channel_id}',")
        print("}")
        
        return True
            
    except Exception as e:
        print(f"✗ Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_youtube_api()
    sys.exit(0 if success else 1) 