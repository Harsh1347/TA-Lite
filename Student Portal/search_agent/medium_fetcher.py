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