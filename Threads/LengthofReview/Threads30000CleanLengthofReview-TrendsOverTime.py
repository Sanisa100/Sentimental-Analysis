import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Threads\\Threads30000\\Threads30000-Data Preprocessing and Cleaning.csv')

# Convert the date column to datetime format with dayfirst=False
df['review_date'] = pd.to_datetime(df['review_date'], dayfirst=False)

# Group by date and get the average review length for that date
df['review_length'] = df['review_description'].str.len()
df_date = df.groupby('review_date')['review_length'].mean().reset_index()

plt.figure(figsize=(15, 5))
sns.lineplot(x='review_date', y='review_length', data=df_date)
plt.title('Trend of Review Length Over Time')
plt.ylabel('Average Review Length')
plt.xlabel('Date')
plt.show()
