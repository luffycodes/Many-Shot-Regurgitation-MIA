import pandas as pd
import re
import sys

def preprocess_string(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    # df = df[df['word_count'] < 1000]
    df = df.head(600)
    # df = df.head(74)
    print("Original number of rows:", df.shape[0])

    # Check if 'contents' column exists and rename it to 'content'
    if 'contents' in df.columns:
        df = df.rename(columns={'contents': 'content'})

    for value in range(4, 16):
        #print(f"\nFiltering for rows with lcs_substring word count > {value}")
        filtered_df = df[df.apply(lambda row: len(row['lcs_substring'].split()) > value, axis=1)].copy()
        filtered_df['preprocessed_contents'] = filtered_df['content'].apply(preprocess_string)
        filtered_df['preprocessed_lcs_substring'] = filtered_df['lcs_substring'].apply(preprocess_string)

        filtered_df = filtered_df[filtered_df.apply(lambda row: row['preprocessed_lcs_substring'] in row['preprocessed_contents'] and row['preprocessed_contents'].count(row['preprocessed_lcs_substring']) == 1, axis=1)]

        print(f"#rows post filter for verbatim match > {value} words:", filtered_df.shape[0])
        arr.append(filtered_df.shape[0])

if __name__ == '__main__':
    # filename = "april_wiki_1k_min/gpt3_1k_fixed_temp_dot7_splits_6_april_wiki_1k_min.csv"
    filename = "../wiki/gpt3_1k_fixed_temp_dot1_splits_6_wikimedia_wikipedia.csv"
    # filename = "../wiki/gpt4_1k_fixed_temp_dot1_splits_6_wikimedia_wikipedia.csv"
    filename = "../wiki/wiki_125/gpt3_125_fixed_temp_dot1_splits_6_wikimedia.csv"
    # filename = "april_wiki_75/gpt3_75_fixed_temp_dot1_splits_6_april_wiki_75.csv"
    arr = []
    process_file(filename)
    print("hey")
    print(arr)
