import requests
import re
from bs4 import BeautifulSoup
import argparse


def fetch_data_from_url(url):
    # Regular expression pattern for a simple URL validation (can be customized)
    pattern = re.compile(r'https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}')

    if not pattern.match(url):
        return "Invalid URL format"

    response = requests.get(url)

    if response.status_code == 200:
        if 'text/html' in response.headers['Content-Type']:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Additional structural checks can be inserted here

            paragraphs = soup.find_all('p')
            extracted_text = " ".join([paragraph.text for paragraph in paragraphs])

            # Check for minimum text length, as an example
            if len(extracted_text) < 100:
                return "Insufficient text length to be considered a news article"

            return extracted_text
        else:
            return "Invalid content type; text or HTML expected"
    else:
        return f"Failed to fetch data from URL. Status code: {response.status_code}"


if __name__ == "__main__":
    # Initialize the argparse class object
    parser = argparse.ArgumentParser(description="Fetch data from a given URL.")

    # Add the argument(s) that you want to parse
    parser.add_argument("url", type=str, help="The URL from which to fetch data.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments and call your function
    fetched_data = fetch_data_from_url(args.url)

    print(fetched_data)