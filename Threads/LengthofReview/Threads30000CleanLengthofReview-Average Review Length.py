import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming your CSV file's name is 'data.csv'
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

# Compute polarity
df['polarity'] = df['review_description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

df['review_length'] = df['review_description'].str.len()
df.dropna(subset=['review_length'], inplace=True)

sns.barplot(x='sentiment', y='review_length', data=df, estimator=np.mean, errorbar=None)

plt.title('Average Review Length by Sentiment')
plt.ylabel('Average Review Length')
plt.xlabel('Sentiment')
plt.show()
