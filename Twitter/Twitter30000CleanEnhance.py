# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from wordcloud import WordCloud
import pandas as pd
from textblob import TextBlob

# Load the cleaned data into a DataFrame
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

# If 'review_description' contains any NaN values, drop those rows
df = df.dropna(subset=['review_description'])

# Conducting sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['review_description'].apply(get_sentiment)

# Visualize the distribution of sentiments
sentiments = df['sentiment'].value_counts()

# 1. Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiments.index, y=sentiments.values, alpha=0.8, palette="viridis")
plt.title('Sentiment Distribution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Sentiment', fontsize=12)
plt.show()

# 2. Average Ratings per Sentiment
average_ratings = df.groupby('sentiment')['rating'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=average_ratings.index, y=average_ratings.values, alpha=0.8, palette="viridis")
plt.title('Average Ratings per Sentiment')
plt.ylabel('Average Rating', fontsize=12)
plt.xlabel('Sentiment', fontsize=12)
plt.show()

# 3. Word Cloud
def generate_wordcloud(text_series, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_series))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Positive word cloud
generate_wordcloud(df[df['sentiment'] == 'positive']['review_description'], 'Word Cloud for Positive Reviews')

# Negative word cloud
generate_wordcloud(df[df['sentiment'] == 'negative']['review_description'], 'Word Cloud for Negative Reviews')
