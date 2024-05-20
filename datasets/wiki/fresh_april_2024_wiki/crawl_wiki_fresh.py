import argparse

import pandas as pd
from openai import OpenAI, OpenAIError

from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import requests
import mwparserfromhell
import re
from urllib.parse import quote

CAT_ALIASES = {}
MEDIA_ALIASES = {}

def _clean_content(raw_content, title, language):
    """Cleans raw wikicode to extract text."""
    try:
        text = _parse_and_clean_wikicode(raw_content, parser=mwparserfromhell, language=language)
    except (mwparserfromhell.parser.ParserError) as e:
        print("mwparserfromhell ParseError: ", e)
        return

    if not text:
        return

    url = _construct_url(title, language)

    return {"url": url, "title": title, "text": text}

def _construct_url(title, language):
    # See: https://meta.wikimedia.org/wiki/Help:URL
    return f"https://{language}.wikipedia.org/wiki/{quote(title)}"

def _parse_and_clean_wikicode(raw_content, parser, language):
    """Strips formatting and unwanted sections from raw page content."""
    wikicode = parser.parse(raw_content)

    # Filters for magic words that are parser instructions -- e.g., __NOTOC__
    re_rm_magic = re.compile("__[A-Z]*__", flags=re.UNICODE)

    # Filters for file/image links.
    media_prefixes = "|".join(["File", "Image", "Media"] + MEDIA_ALIASES.get(language, []))
    re_rm_wikilink = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def rm_wikilink(obj):
        return bool(re_rm_wikilink.match(str(obj.title)))

    # Filters for references and tables
    def rm_tag(obj):
        return str(obj.tag) in {"ref", "table"}

    # Leave category links in-place but remove the category prefixes
    cat_prefixes = "|".join(["Category"] + CAT_ALIASES.get(language, []))
    re_clean_wikilink = re.compile(f"^(?:{cat_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def is_category(obj):
        return bool(re_clean_wikilink.match(str(obj.title)))

    def clean_wikilink(obj):
        text = obj.__strip__()
        text = re.sub(re_clean_wikilink, "", text)
        obj.text = text

    def try_replace_obj(obj):
        try:
            clean_wikilink(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    section_text = []
    # Filter individual sections to clean.
    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        for obj in section.ifilter_wikilinks(recursive=True):
            if rm_wikilink(obj):
                try_remove_obj(obj, section)
            elif is_category(obj):
                try_replace_obj(obj)
        for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
            try_remove_obj(obj, section)

        section_text.append(re.sub(re_rm_magic, "", section.strip_code().strip()))
    return "\n\n".join(section_text)


if __name__ == '__main__':

    result_df = pd.DataFrame()

    index = 0
    # Example usage
    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"

    SEARCHPAGE = "Wikipedia:New pages"

    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rcnamespace": "0",
        "rcprop": "title|ids|sizes",
        "rcshow": "new",
        "rctype": "new",  # Add this parameter to filter only new pages
        "rclimit": "max"
    }

    times_continue = 0

    while True:
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        RECENTCHANGES = DATA['query']['recentchanges']
        for rc in RECENTCHANGES:
            try:
                if rc["type"] != "new" and rc['newlen'] < 10000000:
                    continue

                PARAMS_2 = {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content|timestamp",
                    "format": "json",
                    "titles": rc["title"]
                }

                R2 = S.get(url=URL, params=PARAMS_2)
                DATA2 = R2.json()

                pages = DATA2['query']['pages']
                page_data = list(DATA2['query']['pages'].values())[0]
                if 'revisions' in page_data:
                    page_content = page_data['revisions'][0]['*']
                    page_timestamp = page_data['revisions'][0]['timestamp']  # Get timestamp
                    cleaned_content = _clean_content(page_content, rc["title"], "en")
                else:
                    print(f"No revisions found for URL: {URL}")
                    continue

                if len(cleaned_content["text"].split()) > 500:
                    print({"url": cleaned_content["url"], "title": rc["title"], "page_timestamp": page_timestamp, "length": len(cleaned_content["text"].split())})
                    print(index)
                    index = index + 1
                else:
                    continue

                result_df.loc[index, 'id'] = index
                result_df.loc[index, 'url'] = cleaned_content["url"]
                result_df.loc[index, 'content'] = cleaned_content["text"]
                result_df.loc[index, 'page_timestamp'] = page_timestamp

            except Exception as e:
                print(f"Error: {str(e)}")

            result_df.to_csv(f'new_wiki.csv', index=False)

        result_df.to_csv(f'new_wiki.csv', index=False)

        if 'continue' in DATA:
            PARAMS['rccontinue'] = DATA['continue']['rccontinue']
            times_continue = times_continue + 1
            print("times_continue: ", times_continue)
        else:
            break