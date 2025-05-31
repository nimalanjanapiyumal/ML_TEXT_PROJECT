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
