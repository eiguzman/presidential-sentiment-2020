import numpy as np
import pandas as pd
import re

# Read the TSV file into a DataFrame
df = pd.read_csv("./data/for_comparing.csv")

# Convert all strings in 'CleanText' to lowercase
df['CleanText'] = df['CleanText'].str.lower()

# Define the list of keywords to search for
keywords = ['covid-19', 'vaccin', 'coronavirus', 'corona', 'pandemic']

# Create a regex pattern for the keywords, interpreting '*' as wildcard
pattern = re.compile('|'.join([kw.replace('*', '.*') for kw in keywords]))

# Function to determine if the text contains any of the keywords
def contains_keywords(text):
    return 1 if pattern.search(text) else 0

# Function to count total occurrences of all keywords in the text
def count_keywords(text):
    matches = re.findall(pattern, text)
    return len(matches)

# Apply the functions to create the 'Related' and 'Frequency' columns
df['Related'] = df['CleanText'].apply(contains_keywords)
df['Frequency'] = df['CleanText'].apply(count_keywords)

# Add the 'size' column as log2(2 + Frequency)
df['size'] = round(np.log2(2 + df['Frequency']), 2)

# Save the resulting DataFrame to a TSV file named "pyt.tsv"
df.to_csv("./data/final.tsv", sep='\t', index=False)