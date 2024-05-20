from scipy.stats import kruskal
import scikit_posthocs as sp

from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp


def cliffs_delta(x, y):
    u, p = mannwhitneyu(x, y, method='asymptotic')
    n1, n2 = len(x), len(y)
    delta = (2 * u) / (n1 * n2) - 1
    return delta


lengths = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
start = 1
end = len(lengths)-3
lengths = lengths[start:end]

splits = 6
if splits == 8:
    # Verbatim match lengths and corresponding frequencies
    # gpt3.5 data
    april_wiki_frequencies_gpt3_5 = [71, 46, 21, 16, 11, 7, 7, 7, 6, 6, 5, 4][start:end]
    wikimedia_frequencies_gpt3_5 = [167, 129, 105, 89, 81, 69, 62, 54, 48, 46, 43, 41][start:end]
    # gpt4 data
    april_wiki_frequencies_gpt4 = [54, 23, 12, 7, 4, 3, 3, 3, 3, 3, 3, 3][start:end]
    wikimedia_frequencies_gpt4 = [132, 87, 57, 38, 29, 22, 18, 16, 15, 12, 8, 7][start:end]
    # llama data
    april_wiki_frequencies_llama = [107, 71, 38, 26, 20, 10, 9, 8, 7, 7, 6, 6][start:end]
    wikimedia_frequencies_llama = [158, 108, 77, 52, 40, 29, 22, 14, 10, 8, 7, 6][start:end]

if splits == 6:
    # Verbatim match lengths and corresponding frequencies
    # gpt3.5 data
    april_wiki_frequencies_gpt3_5 = [68, 32, 20, 14, 10, 6, 5, 5, 4, 4, 3, 2][start:end]
    wikimedia_frequencies_gpt3_5 = [195, 153, 124, 108, 97, 87, 80, 74, 69, 66, 62, 57][start:end]
    # gpt4 data
    april_wiki_frequencies_gpt4 = [67, 34, 20, 12, 7, 4, 3, 2, 2, 2, 1, 1][start:end]
    wikimedia_frequencies_gpt4 = [155, 97, 72, 59, 46, 34, 26, 19, 14, 10, 6, 4][start:end]
    # llama data
    april_wiki_frequencies_llama = [116, 74, 45, 26, 18, 11, 10, 7, 5, 4, 4, 4][start:end]
    wikimedia_frequencies_llama = [188, 125, 87, 63, 53, 34, 27, 18, 14, 13, 13, 12][start:end]

datasets = [
    {"name": "GPT3.5", "april": april_wiki_frequencies_gpt3_5, "wikimedia": wikimedia_frequencies_gpt3_5},
    {"name": "GPT4", "april": april_wiki_frequencies_gpt4, "wikimedia": wikimedia_frequencies_gpt4},
    {"name": "Llama", "april": april_wiki_frequencies_llama, "wikimedia": wikimedia_frequencies_llama}
]

for data in datasets:
    print("\nPerforming tests for", data["name"], "data...\n")

    delta = cliffs_delta(data["april"], data["wikimedia"])
    print("Cliff's Delta:", delta)

    ks_statistic, p_value = ks_2samp(data["april"], data["wikimedia"])
    print("KS Distance:", ks_statistic)
    print("p-value:", p_value)

    # Perform the Kruskal-Wallis H test
    h_statistic, p_value = kruskal(data["april"], data["wikimedia"])
    print("Kruskal-Wallis H test:")
    print("H-statistic:", h_statistic)
    print("p-value:", p_value)

    # Perform Dunn's test
    dunn_results = sp.posthoc_dunn([data["april"], data["wikimedia"]], p_adjust='bonferroni')
    print("\nDunn's test results:")
    print(dunn_results)
