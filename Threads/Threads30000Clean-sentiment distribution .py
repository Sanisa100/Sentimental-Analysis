import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
# Load your data into a DataFrame
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')

nltk.download('stopwords')
# Drop rows where 'review_description' is NaN
df = df.dropna(subset=['review_description'])

# Convert all reviews to lowercase for consistency
df['review_description'] = df['review_description'].str.lower()
# Tokenize the review descriptions and remove stopwords
stop_words = set(stopwords.words('english'))
df['tokenized_review'] = df['review_description'].apply(nltk.word_tokenize)
df['tokenized_review'] = df['tokenized_review'].apply(lambda x: [word for word in x if word not in stop_words])
# Function to categorize the polarity
def categorize_polarity(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['polarity'] = df['review_description'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(categorize_polarity)
# Plot sentiment distribution
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.show()
