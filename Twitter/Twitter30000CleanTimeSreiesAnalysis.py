import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the cleaned data into a DataFrame
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

# Convert the date column to the correct datetime format
df['review_date'] = pd.to_datetime(df['review_date'], format='%m/%d/%Y %H:%M')

# Compute polarity of reviews using TextBlob
df['polarity'] = df['review_description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Categorize sentiments based on polarity
def categorize_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['polarity'].apply(categorize_sentiment)

# Group by date and sentiment to get sentiments over time
sentiments_over_time = df.groupby([df['review_date'].dt.date, 'sentiment']).size().unstack().fillna(0)

# Plot sentiments over time
sentiments_over_time.plot(figsize=(15, 7), title='Sentiments Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.show()
