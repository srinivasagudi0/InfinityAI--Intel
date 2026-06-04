# USED this to expand tink
#  Infinity-0 Data Crawler

import os
import re
import time
from bs4 import BeautifulSoup
import requests

# Config 

OUTPUT_PATH = "data/tiny_corpus.txt"
TARGET_CHARS = 15_000 # stop when we hit this many chars
CHUNK_SIZE = 1000 # write this many at a time to disk
SLEEP_BETWEEN = 0.5 # seconds to sleep between requests, so it doesn't look like we're spamming the server
MIN_LINE_LEN = 40 # Skip lines shorter than this, (usually headers, footers, etc)

# Wikipedia topics to crawl
TOPICS = [
    "Artificial_Intelligence", "Machine_Learning", "Neural_Networks",
    "Robots",
    ]

def clean_text(raw):
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) < MIN_LINE_LEN:
            continue

        special_count = sum(1 for c in line if not c.isalnum() and c not in " .,!?;:'-\n")
        if special_count > len(line) * 0.2:   # more than 20% special chars → skip
            continue

        line = re.sub(r' +', ' ', line)  # collapse multiple spaces

        line = re.sub(r'\[.*?\]', '',line )

        line = re.sub(r'\([^)]{40,}\)', '', line)


        if len(line) < MIN_LINE_LEN:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def fetch_wiki(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"

    headers = {
        "User-Agent": "MOZILLA/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            print(f"No content div found for '{topic}'")
            return ""

        paragraphs = content_div.find_all('p')
        raw_text = "\n".join(p.get_text() for p in paragraphs)
        cleaned_text = clean_text(raw_text)

        return cleaned_text
        
    
    except Exception as e:
        print(f"Error fetching '{topic}': {e}")
        return ""
    
def write_chunk(path, text):
    with open(path, 'a') as f:
        for i in range(0, len(text), CHUNK_SIZE):
            f.write(text[i : i + CHUNK_SIZE])
    

def get_current_size(path):
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def crawl():
    os.makedirs('data', exist_ok=True)

    print('Target chars:', TARGET_CHARS)
    print('Topics:', len(TOPICS))
    print('='*30)

    total_written = 0
    articles_done = 0

    for topic in TOPICS:
        #if total_written >= TARGET_CHARS:
        #    print(f"Target reached - {total_written} chars written.")
         #   break

        print(f"Fetching '{topic}'...")
        text = fetch_wiki(topic)
        if not text:
            print(f"No content for '{topic}', skipping.")
            continue

        write_chunk(OUTPUT_PATH, text + "\n\n")

        chars_added = len(text)
        total_written += chars_added
        articles_done += 1

        print(f"+{chars_added:,} chars | total: {total_written:,} chars / {TARGET_CHARS:,} chars" )
        
        print(f"CRAWLED: {topic} | Total articles: {articles_done} | Total chars: {total_written:,}")

        time.sleep(SLEEP_BETWEEN)

    
    print()
    print("="*30)
    print(f"Done. ")
    print(f"Articles crawled: {articles_done}")
    print(f"Total charecters : {total_written:,}")
    print('Saved to :', OUTPUT_PATH)

    actual_size = get_current_size(OUTPUT_PATH)
    print(f"Actual file size: {actual_size:,} chars on disk")
    print(f"Total chars : {total_written:,} chars in memory")
    print(f"Difference (due to encoding, newlines, etc): {actual_size - total_written:,} chars")

    if actual_size >= TARGET_CHARS:
        print("Target MET - ready to retrain Infinity-0!")
    else:
        print(f"Target NOT met - consider adding more topics or increasing target chars. Need {TARGET_CHARS - actual_size:,} more chars.")


if __name__ == "__main__":
    crawl()



