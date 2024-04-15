import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
with open('preprocessed_intents.json', 'r') as file:
    preprocessed_data = json.load(file)

# Extract labels
labels = [intent['tag'] for intent in preprocessed_data]

# Load the trained model
model = load_model('chatbot_model.h5')

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Make predictions
y_pred = model.predict_classes(X_test)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
