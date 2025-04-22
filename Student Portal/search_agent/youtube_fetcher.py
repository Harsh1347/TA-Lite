from googleapiclient.discovery import build
import random

def fetch_youtube_videos(api_key, channel_id, query, max_results=2):
    youtube = build("youtube", "v3", developerKey=api_key)

    search_request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        q=query,
        maxResults=10,
        type="video",
        order="relevance"
    )

    response = search_request.execute()

    videos = []
    for item in response.get("items", []):
        video = {
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        videos.append(video)

    if len(videos) > max_results:
        return random.sample(videos, max_results)

    return videos
