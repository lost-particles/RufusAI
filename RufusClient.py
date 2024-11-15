import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class RufusClient:
    def __init__(self, api_key=None, dynamic_content=False):
        self.api_key = api_key
        self.dynamic_content = dynamic_content
        self.visited_links = set()
        self.extracted_data = []

        # Checking Api key before proceeding
        # Load a pre-trained transformer model for NLP filtering
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Configure Selenium for dynamic content handling if needed
        if self.dynamic_content:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            self.driver = webdriver.Chrome(options=chrome_options)
        else:
            self.driver = None

    def embed_text(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**tokens)
        return output.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embedding

    def compute_similarity(self, query_embedding, text_embedding):
        return torch.nn.functional.cosine_similarity(query_embedding, text_embedding).item()

    def fetch_page(self, url):
        try:
            if self.dynamic_content and self.driver:
                self.driver.get(url)
                time.sleep(3)  # Allow time for content to load
                return self.driver.page_source
            else:
                response = requests.get(url)
                response.raise_for_status()
                return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def crawl_and_extract(self, base_url, user_instruction, max_depth=3):
        self.user_instruction = user_instruction
        self.user_instruction_embedding = self.embed_text(user_instruction)
        self.crawl(base_url, 1, max_depth)

    def crawl(self, url, depth=1, max_depth=3):
        if depth > max_depth or url in self.visited_links:
            return

        self.visited_links.add(url)
        page_content = self.fetch_page(url)
        if not page_content:
            return

        soup = BeautifulSoup(page_content, 'html.parser')
        self.extract_relevant_data(soup, url)

        for link in soup.find_all('a', href=True):
            full_url = self.normalize_url(url, link['href'])
            if self.is_valid_link(full_url):
                self.crawl(full_url, depth + 1, max_depth)

    def normalize_url(self, base_url, url):
        if url.startswith('/'):
            return f"{base_url.rstrip('/')}{url}"
        elif not url.startswith('http'):
            return f"{base_url.rstrip('/')}/{url}"
        return url

    def is_valid_link(self, url):
        return url.startswith('http') and url not in self.visited_links

    def extract_relevant_data(self, soup, source_url):
        relevant_content = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span']):
            text = element.get_text().strip()
            #print(text)
            if text:
                text_embedding = self.embed_text(text)
                similarity_score = self.compute_similarity(self.user_instruction_embedding, text_embedding)
                if similarity_score > 0.7:  # Threshold for relevance
                    print(f'matched content {text}')
                    relevant_content.append({
                        "text": text,
                        "similarity_score": similarity_score
                    })

        if relevant_content:
            self.extracted_data.append({
                "source": source_url,
                "content": relevant_content
            })

    def scrape(self, url, user_instruction, max_depth=3):
        self.crawl_and_extract(url, user_instruction, max_depth)
        return self.get_structured_output()

    def get_structured_output(self, format="json"):
        if format == "json":
            return json.dumps(self.extracted_data, indent=4)
        elif format == "csv":
            import csv
            output = "source,text,similarity_score\n"
            for entry in self.extracted_data:
                for item in entry['content']:
                    output += f"{entry['source']},{item['text'].replace(',', ' ')},{item['similarity_score']}\n"
            return output
        else:
            raise ValueError("Unsupported output format. Use 'json' or 'csv'.")

    def __del__(self):
        if self.driver:
            self.driver.quit()


# Example usage
if __name__ == "__main__":
    import os

    key = os.getenv('Rufus_API_KEY')  # Assume an environment variable for API key (optional)
    client = RufusClient(api_key=key, dynamic_content=True)
    instructions = "Agentic System"
    documents = client.scrape("https://www.withchima.com", instructions, max_depth=2)
    print(documents)
