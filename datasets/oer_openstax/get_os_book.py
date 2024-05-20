import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL of the webpage to scrape
from tqdm import tqdm

def check_start_int_int(s):
    pattern = r'^\d+\.\d+'
    return bool(re.match(pattern, s))


# biology
book_name = "bio_2e_os_book"
url = "https://openstax.org/books/biology-2e/pages/1-1-the-science-of-biology"
url_head = "https://openstax.org/books/biology-2e/pages/"

# concepts biology
url = "https://openstax.org/books/concepts-biology/pages/1-1-themes-and-concepts-of-biology"
url_head = "https://openstax.org/books/concepts-biology/pages/"

# econ 2e
url = "https://openstax.org/books/principles-economics-2e/pages/1-1-what-is-economics-and-why-is-it-important"
url_head = "https://openstax.org/books/principles-economics-2e/pages/"

# nursing
url = "https://openstax.org/books/nutrition/pages/1-1-what-is-nutrition"
url_head = "https://openstax.org/books/nutrition/pages/"

# Send a GET request to the URL and get the webpage content
response = requests.get(url)
content = response.content

# Parse the webpage content using BeautifulSoup
soup = BeautifulSoup(content, "html.parser")

# Find all `li` elements with `data-type` attribute set to `page`
page_items = soup.find_all("li", {"data-type": "page"})

# Extract the text content of each `li` element and print it
non_section_headings = ['Preface', 'Introduction', 'Key Terms', 'Chapter Summary', 'Visual Connection Questions',
                        'Review Questions', 'Critical Thinking Questions',
                        'A | The Periodic Table of Elements',
                        'B | Geological Time',
                        'C | Measurements and the Metric System',
                        'Index', 'Suggested Reading', 'Chapter'
                        ]
section_id = []
section_name = []
section_href = []
section_content_arr = []

for item in tqdm(page_items):
    learning_objs = []
    item_text = item.get_text().strip()
    href = item.a["href"]
    if check_start_int_int(item_text.strip()):
        print(item_text)

        # URL of the webpage to scrape
        url = url_head + href

        # Send a GET request to the URL and get the webpage content
        response2 = requests.get(url)
        content2 = response2.content

        # Find the `p` tag with `id` attribute set to `para-00001`
        soup2 = BeautifulSoup(content2, "html.parser")
        sections = soup2.find_all('section', {'data-depth': '1'})

        section_content = ""
        for section in sections:
            content = section.get_text()
            content = re.sub(r'\n+', '\n', content)
            section_content = section_content + content

        section_content = re.sub(r'\n+', '\n', section_content)

        section_id.append(item_text.split(' ')[0])
        section_name.append(' '.join(item_text.split(' ')[1:]))
        section_href.append(href)
        section_content_arr.append(section_content)
        # print(section_content)

data = {'section_id': section_id, 'section_name': section_name, 'section_url': section_href, 'contents': section_content_arr}
df = pd.DataFrame(data)

# write dataframe to CSV file without index
df.to_csv(f'{book_name}.csv', index=False)
