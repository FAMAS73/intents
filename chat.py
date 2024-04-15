import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('trained_model.keras')

# Function to predict intent and generate response
def get_chatbot_response(user_input):
    # Preprocess user input
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    # Predict intent
    prediction = model.predict(padded_sequence)
    # Get predicted tag
    predicted_tag = unique_tags[np.argmax(prediction)]
    # Get response for predicted tag
    responses_index = tags.index(predicted_tag)
    response = np.random.choice(responses[responses_index])
    return response

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = get_chatbot_response(user_input)
    print("Chatbot:", response)