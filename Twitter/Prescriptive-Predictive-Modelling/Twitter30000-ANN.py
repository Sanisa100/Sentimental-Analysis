import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Loading the data
df = pd.read_csv('C:\\Users\\sanisa\\Documents\\CSIT922\\GrProj\\Twitter\\Twitter30000-Data Preprocessing and Cleaning.csv')

# Calculating sentiment polarity
df['polarity'] = df['review_description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 5 if x > 0.6 else (4 if x > 0.2 else (3 if x > -0.2 else (2 if x > -0.6 else 1))))

# Splitting the data
X = df[['polarity']]  # Features
y = df['sentiment']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# One-hot encoding the target variable for ANN
y_train_cat = to_categorical(y_train - 1, num_classes=5)  # subtracting 1 to make classes start from 0
y_test_cat = to_categorical(y_test - 1, num_classes=5)

# Building the ANN model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=1))  # Input layer
model.add(Dense(10, activation='relu'))  # Hidden layer
model.add(Dense(5, activation='softmax'))  # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train_cat, epochs=50, batch_size=10, verbose=1)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print("\nANN Accuracy:", accuracy)
