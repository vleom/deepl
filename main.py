# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Step 1: Understanding the Problem and Dataset
In this step, we familiarize ourselves with the dataset and the problem to be solved. 
We understand that we need to classify YouTube comments as spam or ham. We note that the TAG attribute in the dataset serves this purpose.
"""

"""
Step 2: Research
In this step, we've read up on spam detection techniques and the use of deep learning in text classification problems.
"""

"""
Step 3: Preprocessing the Data
"""
# Load the dataset from the CSV files
file_names = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv', 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
dfs = []
for file_name in file_names:
    dfs.append(pd.read_csv(file_name))
df = pd.concat(dfs)

# Check the first few rows of the dataframe
print(df.head())

# Understand the structure of the data
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Observe the distribution of spam and not-spam comments
sns.countplot(x='TAG', data=df)
plt.title('Distribution of Spam and Not-Spam Comments')
plt.show()

# Examine some examples of spam and not-spam comments
print("Spam comments:")
print(df[df['TAG']==1]['CONTENT'].head())
print("\nNot-Spam comments:")
print(df[df['TAG']==0]['CONTENT'].head())

# Tokenize the text
# fit_on_texts Updates internal vocabulary based on a list of texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.CONTENT)

# text_to_sequences Transforms each text in texts to a sequence of integers
sequences = tokenizer.texts_to_sequences(df.CONTENT)

# pad_sequences is used to ensure that all sequences in a list have the same length.
# By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.
data = pad_sequences(sequences)

# One-hot encode the labels
# Converting the label to binary array representation
labels = to_categorical(np.asarray(df.TAG))

"""
Step 4: Designing the Deep Learning Model
"""
# The model architecture is built here. We're using an embedding layer, an LSTM layer with dropout, and a Dense output layer with softmax activation.
model = Sequential()
model.add(Embedding(10000, 128, input_length=data.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

"""
Step 5: Training the Model
"""
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Compile the model
# We're using the RMSprop optimizer and the binary cross-entropy loss function,
# because this is a binary classification problem. We're also tracking accuracy, precision, and recall as metrics.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

"""
Step 6: Evaluating the Model
"""
# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy, precision and recall
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
precision = precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
recall = recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

print('Accuracy: %f' % (accuracy*100))
print('Precision: %f' % (precision*100))
print('Recall: %f' % (recall*100))

"""
Step 7: Implementing Transfer Learning
In this step, we would select an appropriate pre-trained model and implement transfer learning by using the pre-trained model 
and adding additional layers if necessary. We would then train and evaluate this new model, comparing the results with our original model.

Step 8: Improving the Model
Based on the results, we identify possible areas of improvement. This could be in the model architecture, the data preprocessing step, or others.
We implement these improvements and re-evaluate the model. We continue this trial-and-error process until satisfactory results are obtained.
"""

"""
Step 9: Documentation and Presentation
Prepare a two-page document summarizing your approach and results.
Write the code cleanly and include comments for better understanding.
Prepare the presentation slides. Be sure to clearly explain the problem, your approach, the model architecture, your findings, 
and any insights or interesting things you learned from the project.
Practice presenting your findings, anticipating possible questions that may be asked, and preparing answers for them.
"""

"""
Step 10: Reflection
Reflect on the project, the methods used, and the results obtained.
Identify what went well and what could be improved.
Consider how the skills learned in this project could be applied to other projects in the future.
"""
