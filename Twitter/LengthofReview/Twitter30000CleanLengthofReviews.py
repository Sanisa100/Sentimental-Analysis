import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the CSV file into the DataFrame 'df'
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

from textblob import TextBlob

df['polarity'] = df['review_description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Now you can perform operations on 'df'
df['review_length'] = df['review_description'].str.len()

# Histogram
plt.figure(figsize=(15, 7))
sns.histplot(df, x="review_length", hue="sentiment", element="step", stat="probability", common_norm=False)
plt.title('Distribution of Review Length by Sentiment')
plt.xlabel('Review Length')
plt.ylabel('Probability')
plt.show()

print_top_words(lda_model, vectorizer.get_feature_names_out())
