import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from textblob import TextBlob
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import pydot
import webbrowser
import os

# Loading the data
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

# Calculating sentiment polarity
df['polarity'] = df['review_description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 5 if x > 0.6 else (4 if x > 0.2 else (3 if x > -0.2 else (2 if x > -0.6 else 1))))

# Splitting the data
X = df[['polarity']]  # Features
y = df['sentiment']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Evaluating the model
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nPrecision (Macro):", precision_score(y_test, predictions, average='macro'))
print("\nRecall (Macro):", recall_score(y_test, predictions, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Visualizing the Decision Tree
dot_data = export_graphviz(clf, out_file=None, feature_names=['polarity'], class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data)[0]
graph.write_png("decision_tree.png", prog='C:\\Program Files\\Graphviz\\bin\\dot.exe')

# ... [rest of your code]

# After saving the decision tree image
file_path = os.path.abspath("decision_tree.png")
webbrowser.open('file://' + file_path)

