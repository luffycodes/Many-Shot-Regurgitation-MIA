import pandas as pd

# Load the CSV file
result_df = pd.read_csv('new_wiki.csv')

# Calculate the average length of content by splitting content into words
result_df['word_count'] = result_df['content'].apply(lambda x: len(x.split()))
average_word_count = result_df['word_count'].mean()
print(f'Average length of content (in words): {average_word_count}')

# Analyze the content column
references_count = 0
before_references_lengths = []
after_references_lengths = []
for content in result_df['content']:
    if 'References' in content:
        references_count += 1
        parts = content.split('References')
        before_references_lengths.append(len(parts[0].split()))
        after_references_lengths.append(len(parts[1].split()))

print(f'Number of contents with "References": {references_count}')
print(f'Average length of content before "References": {sum(before_references_lengths) / len(before_references_lengths)}')
print(f'Average length of content after "References": {sum(after_references_lengths) / len(after_references_lengths)}')

# Analyze the page_timestamp column
result_df['page_timestamp'] = pd.to_datetime(result_df['page_timestamp'])
result_df['month'] = result_df['page_timestamp'].dt.strftime('%Y-%m')
month_counts = result_df['month'].value_counts()
# print(month_counts)

# Trimming
def trim_content(content):
    if 'References' in content:
        return content.split('References')[0]
    else:
        return content


result_df['trimmed_content'] = result_df['content'].apply(trim_content)

april_2024_articles = result_df[result_df['month'] == '2024-04']
april_2024_articles_trimmed = april_2024_articles[april_2024_articles['trimmed_content'].str.split().apply(len) > 1000]
word_counts = april_2024_articles_trimmed['trimmed_content'].str.split().apply(len)
average_word_count = word_counts.mean()
print(f'Average word count april_2024_articles_trimmed: {average_word_count:.2f}')

april_2024_articles_trimmed.drop('content', axis=1, inplace=True)
april_2024_articles_trimmed.rename(columns={'trimmed_content': 'content'}, inplace=True)

april_2024_articles_trimmed.to_csv('april_2024_articles_1k_min_trimmed_correct.csv', index=False)
print("hey")

