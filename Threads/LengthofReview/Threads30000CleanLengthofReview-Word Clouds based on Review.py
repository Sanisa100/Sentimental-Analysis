import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')

from wordcloud import WordCloud
df['review_length'] = df['review_description'].str.len()

# Separate reviews into short and long based on a threshold, e.g., 100 characters
short_reviews = ' '.join(df[df['review_length'] < 100]['review_description'].tolist())
long_reviews = ' '.join(df[df['review_length'] >= 100]['review_description'].tolist())

# Generate word clouds
wordcloud_short = WordCloud(width=800, height=400, background_color='white').generate(short_reviews)
wordcloud_long = WordCloud(width=800, height=400, background_color='white').generate(long_reviews)

# Plotting the word clouds
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_short, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Short Reviews')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_long, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Long Reviews')

plt.show()
