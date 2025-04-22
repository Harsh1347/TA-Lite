import feedparser
import urllib.request
from urllib.parse import quote  

def fetch_arxiv_papers(topic, max_results=5):
    safe_topic = quote(topic)  
    base_url = f"http://export.arxiv.org/api/query?search_query=all:{safe_topic}&start=0&max_results={max_results}"

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    try:
        feed = feedparser.parse(base_url)
    except Exception as e:
        print("Error fetching from arXiv:", e)
        return []  

    papers = []
    for entry in feed.entries:
        papers.append({
            "title": entry.title,
            "summary": entry.summary[:200] + "...",
            "url": entry.link
        })
    return papers
if __name__ == "__main__":
    topic = "Neural Networks"
    results = fetch_arxiv_papers(topic)

    print(f"\n Total papers found: {len(results)}")  

    if not results:
        print(" No papers found. Try a different topic or check network.")
    else:
        for paper in results:
            print(f"- {paper['title']}\n   {paper['url']}\n")

