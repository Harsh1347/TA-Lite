TA-LLM (External Content Agent)


This is a Python-based content agent for our TA project. It fetches educational resources based on any topic you provide.

 What it does:
- Pulls relevant YouTube videos from StatQuest, 3Blue1Brown, and freeCodeCamp
- Fetches recent research papers from arXiv
- (Optionally) Tries to get Medium articles based on the topic

 Files included:
- main.py                  ← Main script to run the full agent
- youtube_fetcher.py       ← YouTube content fetch module
- arxiv_fetcher.py         ← arXiv research paper fetch module
- medium_fetcher.py        ← Medium article fetch module
- requirements.txt         ← List of Python libraries needed
- test_youtube.py          ← Run this to test only YouTube video fetcher
- test_arxiv.py            ← Run this to test only arXiv paper fetcher
- test_medium.py           ← Run this to test only Medium fetcher (may not always work)

To test modules individually:
   test_youtube.py     → tests YouTube search
    test_arxiv.py       → tests arXiv paper fetching
   test_medium.py      → tests Medium article search (can fail sometimes)



