import pandas as pd
import re

# Load the data
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')


# Define a function to check if a given text is meaningful
def is_meaningful(text):
    # Count alphabetic and numeric characters
    alpha_count = sum(c.isalpha() for c in text)
    digit_count = sum(c.isdigit() for c in text)

    # If text is primarily non-alphabetic or is too short, it's considered non-meaningful
    if alpha_count < 0.5 * len(text) or len(text) < 3:
        return False
    return True


# Filter the dataframe for meaningful descriptions
df_cleaned = df[df['review_description'].apply(is_meaningful)]

# Save the cleaned dataframe
df_cleaned.to_csv('cleaned_data.csv', index=False)
