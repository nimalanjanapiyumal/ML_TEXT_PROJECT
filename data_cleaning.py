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
