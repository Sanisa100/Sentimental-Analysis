import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter30000-Data Preprocessing and Cleaning.csv')

df['review_length'] = df['review_description'].str.len()

# Other lines of code...


longest_reviews = df[df['review_length'] > df['review_length'].quantile(0.95)]
print(longest_reviews['review_description'])
