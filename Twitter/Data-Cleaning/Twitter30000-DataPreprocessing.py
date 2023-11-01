import pandas as pd
import re

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000.csv')

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Check for missing values in each column
print("\nMissing values count per column:")
print(df.isnull().sum())

# Fill missing values in 'appVersion' with its mode
mode_appVersion = df['appVersion'].mode()[0]
df['appVersion'].fillna(mode_appVersion, inplace=True)

# Convert 'review_date' column to datetime format with flexible date format
df['review_date'] = pd.to_datetime(df['review_date'], infer_datetime_format=True)

# Filter rows to only include data from June and August 2023
df = df[(df['review_date'].dt.year == 2023) & (df['review_date'].dt.month.isin([6, 7, 8]))]

# Clean review_description
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuations
    text = re.sub('[^a-z0-9]', ' ', text)
    # Tokenize (split the text into words) and join back
    text = ' '.join(text.split())
    return text

df['review_description'] = df['review_description'].apply(clean_text)

# Trim and filter whitespace entries in review_description
df['review_description'] = df['review_description'].str.strip()
df = df[df['review_description'] != ""]

# Handle common indicators of missing data
missing_indicators = ["N/A", "na", "-", "none", "missing"]
df['review_description'].replace(missing_indicators, pd.NA, inplace=True)
df.dropna(subset=['review_description'], inplace=True)

# Re-check for missing values
print("\nMissing values count after cleaning:")
print(df.isnull().sum())

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)
