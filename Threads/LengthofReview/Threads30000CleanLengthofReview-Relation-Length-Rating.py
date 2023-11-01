import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')

# Create a new column for review length
df['review_length'] = df['review_description'].str.len()

plt.figure(figsize=(10, 5))
sns.boxplot(x='rating', y='review_length', data=df)
plt.title('Review Length by Rating')
plt.ylabel('Review Length')
plt.xlabel('Rating')
plt.show()
