import wfdb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def getLabel(value):
    # Your implementation logic here to determine the label based on the given value
    if value == 'N':
        return 0
    elif value == 'V':
        return 1
    elif value == 'S':
        return 2
    elif value == 'F':
        return 3
    else:
        return 4

myDataset = []
heartbeats = []
labels = []  # Updated to include labels

# Extract data and labels
data = []
# Set the path to the directory where the dataset is stored
data_path = '/Users/shamistanhuseynov/PycharmProjects/pythonProject/mit-bih-arrhythmia-database-1.0.0'

# Get a list of all record names in the directory
record_list = [f.replace('.hea', '') for f in os.listdir(data_path) if f.endswith('.hea')]

# Create an empty DataFrame to store the data
df = pd.DataFrame(columns=['Heartbeat', 'Annotation'])

# Loop through each record and load it
for record_name in record_list:
    record_path = os.path.join(data_path, record_name)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    # Extract individual heartbeats
    window_size = 300  # Choose an appropriate window size

    for i in range(0, len(annotation.symbol)):
        if annotation.symbol[i] in ['N', 'V', 'S', 'F', 'Q']:
            center = annotation.sample[i]
            window_start = max(0, center - window_size // 2)
            window_end = min(len(record.p_signal), center + window_size // 2)
            heartbeat = tuple(record.p_signal[window_start:window_end, 0])  # Convert to tuple

            # Check if the heartbeat has the expected length
            if len(heartbeat) == window_size:
                heartbeats.append(heartbeat)
                labels.append(getLabel(annotation.symbol[i]))  # Update labels list

# Convert data and labels to NumPy arrays
data = np.array(heartbeats)
data = data.reshape((data.shape[0], data.shape[1], 1))
labels = np.array(labels)
print(data[0])
print(":")
print(labels[0])
#
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)



model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=300))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with a specified number of epochs
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

model.save("classification_model_v2.keras")
# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)
