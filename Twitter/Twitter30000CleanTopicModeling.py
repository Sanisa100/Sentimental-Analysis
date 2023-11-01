from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataframe
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter30000-Data Preprocessing and Cleaning.csv')

# Calculate polarity and subjectivity (and any other preprocessing steps)
df['polarity'] = df['review_description'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['review_description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Now, perform the vectorization
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
data_vectorized = vectorizer.fit_transform(df['review_description'])

# Vectorize the reviews
vectorizer = CountVectorizer(max_df=0.9, min_df=25, stop_words='english')
data_vectorized = vectorizer.fit_transform(df['review_description'])

# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_Z = lda_model.fit_transform(data_vectorized)

# Print top words for each topic
def print_top_words(model, feature_names, n_words=10):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_words - 1:-1]])
        print(message)

print_top_words(lda_model, vectorizer.get_feature_names_out())
