import requests
from bs4 import BeautifulSoup

def fetch_medium_articles(topic, max_results=5):
    url = f"https://medium.com/search?q={topic.replace(' ', '%20')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")

    links = []
    articles = soup.find_all("a", href=True)

    for link in articles:
        href = link["href"]
        if "/p/" in href and len(links) < max_results:
            title = link.get_text(strip=True)
            links.append({
                "title": title if title else "Medium Article",
                "url": href.split("?")[0]
            })

    return links

#testing 
if __name__ == "__main__":
    topic = "Neural Networks"
    results = fetch_medium_articles(topic)

    print("\n Medium Articles on:", topic)
    for article in results:
        print(f"- {article['title']}\n   {article['url']}")
