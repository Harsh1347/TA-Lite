#from youtube_fetcher import fetch_youtube_videos
#from medium_fetcher import fetch_medium_articles
#from arxiv_fetcher import fetch_arxiv_papers
from googleapiclient.discovery import build
import random
import requests

def fetch_medium_articles(topic, serpapi_key, max_results=5):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"site:medium.com {topic}",
        "api_key": serpapi_key
    }

    response = requests.get(url, params=params)
    results = response.json()

    articles = []
    if "organic_results" in results:
        for result in results["organic_results"][:max_results]:
            articles.append({
                "title": result.get("title", "Medium Article"),
                "url": result.get("link", "")
            })
    return articles
from urllib.parse import quote
import feedparser

def fetch_arxiv_papers(topic, max_results=5):
    safe_topic = quote(topic)
    base_url = f"http://export.arxiv.org/api/query?search_query=all:{safe_topic}&start=0&max_results={max_results}"

    feed = feedparser.parse(base_url)

    return [
        {
            "title": entry.title,
            "summary": entry.summary[:200] + "...",
            "url": entry.link
        }
        for entry in feed.entries
    ]

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

def run_agent(query, youtube_api_key, serpapi_key):
    print(f"\nSearching educational content for: {query}\n")
    youtube_api_key = "AIzaSyDigU-C3rQQbBrJ_zeezafCEOtZefJQhBA"
    serpapi_key = "08b8cf3b800b7c953e4d0464b23b7db8bcfdefb1e08f5fa723365cb93b42e8ef"
    

    channels = {
        "StatQuest": "UCtYLUTtgS3k1Fg4y5tAhLbw",
        "3Blue1Brown": "UCYO_jab_esuFRV4b17AJtAw",
        "freeCodeCamp": "UC8butISFwT-Wl7EV0hUK0BQ"
    }

    for name, cid in channels.items():
        print(f"YouTube from {name}:")
        videos = fetch_youtube_videos(youtube_api_key, cid, query)
        if not videos:
            print("  No videos found.")
        else:
            for video in videos:
                print(f"- {video['title']}\n  {video['url']}")

    print(f"\nMedium Articles:")
    articles = fetch_medium_articles(query, serpapi_key)
    if not articles:
        print("  No articles found.")
    else:
        for article in articles:
            print(f"- {article['title']}\n  {article['url']}")

    print(f"\nResearch Papers:")
    papers = fetch_arxiv_papers(query)
    if not papers:
        print("  No research papers found.")
    else:
        for paper in papers:
            print(f"- {paper['title']}\n  {paper['url']}\n")

