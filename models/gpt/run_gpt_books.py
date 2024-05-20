import argparse
import time

import pandas as pd
from openai import OpenAI, OpenAIError


def analyze_strings(result_df, index_id, original_last_message, gpt_response):
    # Longest Common Subsequence
    lcs = longest_common_subsequence(original_last_message, gpt_response)
    result_df.loc[index_id, 'lcs'] = lcs
    overlap_percentage = (len(lcs) / len(original_last_message)) * 100
    result_df.loc[index_id, 'overlap_percentage'] = overlap_percentage

    # Longest Common Subsequence of Words
    lcs_words = longest_common_subsequence_words(original_last_message, gpt_response)
    result_df.loc[index_id, 'lcs_words'] = lcs_words
    lcs_words_percentage = (len(lcs_words.strip().split()) / len(original_last_message.strip().split())) * 100
    result_df.loc[index_id, 'lcs_words_percentage'] = lcs_words_percentage

    # Longest Common Substring
    lcs_substring = longest_common_substring(original_last_message, gpt_response)
    result_df.loc[index_id, 'lcs_substring'] = lcs_substring
    lcs_substring_percentage = (len(lcs_substring.strip().split()) / len(original_last_message.strip().split())) * 100
    result_df.loc[index_id, 'lcs_substring_percentage'] = lcs_substring_percentage

    print(f"\nSection: {result_df.loc[index_id, 'section_url']}")
    print(f"\nLast user message: {original_last_message}")
    print(f"Length of Last user message: {len(original_last_message.split())}")
    print(f"\nGPT response: {gpt_response}")
    print(f"Length of GPT response: {len(gpt_response.split())}")
    # print(f"\nLCS characters: {lcs}")
    # print(f"\nLCS characters Overlap: {overlap_percentage:.2f}%")
    # print(f"\nLCS words: {lcs_words}")
    # print(f"\nLCS words Overlap: {lcs_words_percentage:.2f}%")
    print(f"\nLCS substring: {lcs_substring}")
    print(f"LCS substrings Overlap: {lcs_substring_percentage:.2f}%")

    return result_df


def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    start_pos = end_pos - max_length
    return str1[start_pos:end_pos]


def longest_common_subsequence_words(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    m = len(words1)
    n = len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_words = []
    i, j = m, n
    while i > 0 and j > 0:
        if words1[i - 1] == words2[j - 1]:
            lcs_words.insert(0, words1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ' '.join(lcs_words)


def split_text_into_parts(text, n):
    if n % 2 != 0:
        raise ValueError("n must be an even number")

    words = text.strip().split()
    words_per_part = len(words) // n
    parts = []

    for i in range(n):
        start_index = i * words_per_part
        end_index = (i + 1) * words_per_part

        if i == n - 1:
            end_index = len(words)

        part = ' '.join(words[start_index:end_index])
        parts.append(part)

    return parts


def run_gpt4(prompt, engine, max_tokens=1024, temperature=0.7):
    try:
        response = client.chat.completions.create(model=engine, messages=prompt, max_tokens=max_tokens,
                                                  temperature=temperature)
    except (OpenAIError) as exc:
        print(f"OpenAI error occurred: {exc}, retrying...")
        return "FAIL"
    return response.choices[0].message.content.strip()


def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = ""
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs = str1[i - 1] + lcs
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo-1106",
                        help="engines of the GPT3 model for training")
    # parser.add_argument("--engine", type=str, default="gpt-3.5-turbo-1106", help="engines of the GPT3 model for training")
    parser.add_argument("--max_tokens", type=int, default=1024, help="maximum number of token generated")
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for generation (higher=more diverse)')
    parser.add_argument('--top_p', type=int, default=1, help='p value for sampling')
    parser.add_argument("--n", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--best_of", type=int, default=1, help="Best of N samples to generate")

    params = parser.parse_args()
    return params


if __name__ == '__main__':
    args = add_params()
    client = OpenAI(api_key="")

    # Example usage
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl."

    n = 6  # Number of even parts
    book_name = ""
    folder_name = ""

    df = pd.read_csv(f'{folder_name}/{book_name}')
    result_df = df.copy()

    for index, row in result_df.iterrows():
        print(index)
        # if not pd.isna(result_df.loc[index, 'splits']):
        #     continue
        try:
            parts = split_text_into_parts(str(row["contents"]), n)

            conversation = [{"role": "system", "content": "complete the paragraph"}]
            for i, part in enumerate(parts, 1):
                if i % 2 != 0:
                    conversation.append({"role": "user", "content": part})
                else:
                    conversation.append({"role": "assistant", "content": part})

            last_user_message = conversation[-1]["content"]
            gpt_response = run_gpt4(conversation[:-1], args.engine, args.max_tokens, args.temp)
            result_df.loc[index, 'splits'] = n
            result_df.loc[index, 'last_user_message'] = last_user_message
            result_df.loc[index, 'gpt_response'] = gpt_response

            analyze_strings(result_df, index, last_user_message, gpt_response)
        except ValueError as e:
            print(f"Error: {str(e)}")
            result_df.loc[index, 'lcs'] = "FAIL"
            result_df.loc[index, 'overlap_percentage'] = -1.0
            result_df.loc[index, 'lcs_words'] = "FAIL"
            result_df.loc[index, 'lcs_words_percentage'] = -1.0
            result_df.loc[index, 'lcs_substring'] = "FAIL"
            result_df.loc[index, 'lcs_substring_percentage'] = -1.0

        result_df.to_csv(f'{folder_name}/gpt3_temp_dot{str(round(args.temp*10))}_splits_{str(n)}_{book_name}', index=False)
    result_df.to_csv(f'{folder_name}/gpt3_temp_dot{str(round(args.temp*10))}_splits_{str(n)}_{book_name}', index=False)
