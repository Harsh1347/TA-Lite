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
