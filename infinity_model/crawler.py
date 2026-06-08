import os
import re
import time
import requests

OUTPUT_PATH ="data/tiny_corpus.txt"
TARGET_CHARS = 500_000
CHUNK_SIZE = 500
SLEEP_BETWEEN = 1.0
MIN_LINE_LEN = 40
TOPICS = [
    # Science
    "Photosynthesis", "Gravity", "Evolution", "Cell_biology",
    "Atom", "Electricity", "Magnetism", "Solar_system", "Black_hole",
    "Climate_change", "Ecosystem", "DNA", "Protein", "Neuroscience",
    # Technology
    "Artificial_intelligence", "Computer", "Internet", "Robot",
    "Machine_learning", "Programming_language", "Algorithm",
    # Language & knowledge
    "Language", "Writing", "Mathematics", "Logic", "Philosophy",
    "History", "Library", "Education",
    # Nature
    "Ocean", "River", "Mountain", "Forest", "Weather",
    "Water", "Fire", "Tree", "Bird", "Cat", "Dog",
    # Society
    "Democracy", "Economy", "Agriculture", "Medicine", "Music",
    "Human_brain", "Memory", "Learning", "Communication",
]

def clean_text(raw):
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < MIN_LINE_LEN:
            continue
        # Decode HTML entities first
        line = re.sub(r'&#\d+;', '', line)  # Remove numeric entities like &#93;
        line = re.sub(r'&[a-z]+;', '', line)  # Remove named entities like &nbsp;
        line = re.sub(r'&[#\w]*(?!;)', '', line)  # Remove incomplete entities
        
        special = sum(1 for c in line if not c.isalnum() and not c in ".,!?;:'-\n")
        if special > len(line) * 0.2:
            continue
        line = re.sub(r' +', ' ', line)
        line = re.sub(r'\[.*?\]', '', line)
        line = re.sub(r'\([^)]{40,}\)', '', line)
        line = line.strip()
        if len(line) < MIN_LINE_LEN:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def fetch_wikipedia(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' # this header (the line) is written by AI.
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        # Extract main content between <p> tags
        content = re.findall(r'<p>(.*?)</p>', html, re.DOTALL)
        text = ' '.join(content)
        text = re.sub(r'<[^>]+>', '', text)
        return clean_text(text)
    except Exception as e:
        print(f"{topic} FAILED: {e}")
        return ""

def get_file_chars(path):
    if not os.path.exists(path):
        return 0
    with open(path) as p:
        return len(p.read())
        
def crawl():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print("Cleared old corpus")

    print(f"TARGET: {TARGET_CHARS} chars")
    print(f"TOPICS: {len(TOPICS)}")
    
    total = 0
    done = 0
    for topic in TOPICS:
        if total >= TARGET_CHARS:
            break
        print(f"Fetching {topic}...")
        
        text =fetch_wikipedia(topic)
        if not text:
            print(f"{topic} had no content, skipping.")
            time.sleep(SLEEP_BETWEEN)
            continue
        with open(OUTPUT_PATH, "a") as f:
            for i in range(0, len(text), CHUNK_SIZE):
                f.write(text[i:i+CHUNK_SIZE] + "\n")
            f.write("\n")

        total += len(text)
        done += 1

        print(f" +{len(text):,} | Total: {total:,}")
        time.sleep(SLEEP_BETWEEN)

        
    print()
    print("=="*50)
    actual = get_file_chars(OUTPUT_PATH)
    print(f"Articles: {done}")
    print(f"Total chars: {actual:,}")
    print("Ready to train!" if actual >= TARGET_CHARS else "Not quite there, consider adding more topics.")

if __name__ == "__main__":
    crawl()

