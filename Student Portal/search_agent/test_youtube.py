from googleapiclient.discovery import build

def fetch_youtube_videos(api_key, channel_id, query, max_results=3):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        q=query,
        type="video",
        order="relevance"
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        video_title = item["snippet"]["title"]
        videos.append((video_title, video_url))
    
    return videos


api_key = "AIzaSyDigU-C3rQQbBrJ_zeezafCEOtZefJQhBA"
channel_id = "UCtYLUTtgS3k1Fg4y5tAhLbw"  
query = "Neural Networks"

results = fetch_youtube_videos(api_key, channel_id, query)

print("\n Top YouTube Results from StatQuest:\n")
for title, url in results:
    print(f"- {title}\n   {url}")
