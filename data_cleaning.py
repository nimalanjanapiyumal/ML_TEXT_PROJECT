# data_cleaning.py

import pandas as pd

# Load the Excel files into DataFrames (assuming the first sheet contains the data)
df1 = pd.read_excel('FeedbacksDetails.xlsx')        # First dataset
df2 = pd.read_excel('FeedbacksDetailsBatch_16_17.xlsx')  # Second dataset

# Inspect the first few rows and columns of each to verify loading
print("Dataset 1 columns:", df1.columns.tolist())
print("Dataset 2 columns:", df2.columns.tolist())
print(df1.head(3))  # show first 3 rows of first dataset
print(df2.head(3))  # show first 3 rows of second dataset

# Check basic info and structure
print(df1.info())   # Column names, non-null counts, data types for dataset1
print(df2.info())   # Column names, non-null counts, data types for dataset2

# If needed, describe numeric columns to see ranges, etc.
print(df1.describe(include='all'))
print(df2.describe(include='all'))

# Standardize column names: lowercase and replace spaces/punctuations with underscores
import re

def standardize_columns(df):
    df_copy = df.copy()
    new_cols = []
    for col in df_copy.columns:
        col_clean = col.strip()                       # remove leading/trailing whitespace
        col_clean = col_clean.lower()                 # make lowercase for consistency
        col_clean = re.sub(r'[^\w\s]', '_', col_clean) # replace punctuation with underscores
        col_clean = re.sub(r'\s+', '_', col_clean)     # replace spaces (or multiple spaces) with single underscore
        new_cols.append(col_clean)
    df_copy.columns = new_cols
    return df_copy

# Apply to both dataframes
df1 = standardize_columns(df1)
df2 = standardize_columns(df2)

print("Normalized columns for Dataset 1:", df1.columns.tolist())
print("Normalized columns for Dataset 2:", df2.columns.tolist())

# Manually align any columns that represent the same thing but still have mismatched names (if any)
rename_map_df2 = {}
# Example: if df1 has 'feedback_text' and df2 has 'feedback', we rename df2's 'feedback' -> 'feedback_text'
if 'feedback' in df2.columns and 'feedback_text' in df1.columns:
    rename_map_df2['feedback'] = 'feedback_text'
# (Add other mappings as needed based on inspection)

df2 = df2.rename(columns=rename_map_df2)

# After renaming, ensure both have the same set of columns
for col in df1.columns:
    if col not in df2.columns:
        df2[col] = pd.NA  # add missing column to df2
for col in df2.columns:
    if col not in df1.columns:
        df1[col] = pd.NA  # add missing column to df1

# Re-order columns in df2 to match df1 (optional, for easier comparison)
df2 = df2[df1.columns]

print("Final aligned columns:", df1.columns.tolist())

# Identify columns with too many missing values and drop them
threshold = 0.5  # e.g., drop columns with more than 50% missing values
n_rows1 = len(df1)
cols_to_drop = []
for col in df1.columns:
    missing_frac = df1[col].isna().mean()  # fraction of missing in df1
    if missing_frac > threshold:
        cols_to_drop.append(col)
# (Do the same for df2; or since schemas aligned, use df2 as well)
n_rows2 = len(df2)
for col in df2.columns:
    missing_frac = df2[col].isna().mean()
    if missing_frac > threshold and col not in cols_to_drop:
        cols_to_drop.append(col)

# Drop identified columns from both DataFrames
df1.drop(columns=cols_to_drop, inplace=True)
df2.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns due to missing data > {threshold*100}%:", cols_to_drop)

# Identify and drop constant columns (with only one unique value)
const_cols = [col for col in df1.columns if df1[col].nunique() <= 1]
# (nunique() <= 1 catches both constant and possibly fully NA columns)
df1.drop(columns=const_cols, inplace=True)
df2.drop(columns=const_cols, inplace=True)
print("Dropped constant/irrelevant columns:", const_cols)

# Identify other irrelevant columns to drop (by manual reasoning, e.g., IDs)
irrelevant_cols = []
for col in df1.columns:
    # Example criteria: columns that contain 'id' or 'email' in name, or any personal identifier
    if 'id' in col or 'email' in col:
        irrelevant_cols.append(col)
# Drop them
df1.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
df2.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
print("Dropped explicitly irrelevant columns:", irrelevant_cols)

# Final check of remaining columns
print("Remaining columns after cleanup:", df1.columns.tolist())


# text_preprocessing.py

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # optional, for WordNet lemmatizer to get word meanings

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and tokenize text into normalized tokens."""
    if pd.isna(text):
        return ""  # return empty string for missing text
    # 1. Lowercase the text
    text = text.lower()
    # 2. Remove URLs (if any) and special characters/punctuation
    text = re.sub(r'https?://\S+', ' ', text)        # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)      # keep only letters, numbers, whitespace
    # 3. Tokenize into words
    tokens = word_tokenize(text)  # splits text into tokens (words and punctuation)
    # 4. Remove stopwords and any leftover non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    # 5. Lemmatize each token to its root form
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Return the cleaned tokens as a single space-separated string (or keep as list if preferred)
    return " ".join(tokens)

# Identify the text column(s) to preprocess. Assume 'feedback_text' is the main text field after schema normalization.
text_columns = []
for col in df1.columns:
    # Heuristic: choose object-type columns or known text columns (e.g., ones containing 'feedback' or 'comments')
    if df1[col].dtype == object:
        if 'feedback' in col or 'comment' in col or 'text' in col:
            text_columns.append(col)
print("Text columns identified for preprocessing:", text_columns)

# Apply preprocessing to each identified text column in both datasets
for col in text_columns:
    df1[col + "_clean"] = df1[col].apply(preprocess_text)
    df2[col + "_clean"] = df2[col].apply(preprocess_text)

# Example: if 'feedback_text' is identified, new column 'feedback_text_clean' will have the processed text.
print(df1[['feedback_text', 'feedback_text_clean']].head(5))
