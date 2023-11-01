import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

# Create a new column for review length
df['review_length'] = df['review_description'].str.len()

# Plotting a histogram of review lengths
plt.figure(figsize=(10, 5))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()
