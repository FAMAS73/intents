import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and preprocess the dataset
with open("preprocessed_dataset.json", "r") as file:
    data = json.load(file)

# Extract patterns and intents
patterns = []
intents = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        intents.append(intent["tag"])

# Step 2: Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1
max_seq_length = max([len(seq.split()) for seq in patterns])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Convert intents to numerical labels
label2idx = {label: idx for idx, label in enumerate(np.unique(intents))}
y = np.array([label2idx[label] for label in intents])

# Step 3: Build the model architecture with TensorFlow Lite-compatible operations
# Step 3: Build the model architecture
model = Sequential()
model.add(Input(shape=(max_seq_length,)))
model.add(Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True))
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation="softmax"))

model.summary()

# Step 4: Compile the model
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Step 7: Save the trained model
model.save("intent_model.h5")

# Save the tokenizer
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())


# Step 8: Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Disable lowering tensor list ops
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('intent_model.tflite', 'wb') as f:
    f.write(tflite_model)




# Step 9: Test the TensorFlow Lite model with a responsive chat
def predict_intent_tflite(text):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post')
    interpreter.set_tensor(input_details[0]['index'], padded_sequence)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(prediction)
    for label, idx in label2idx.items():
        if idx == predicted_label:
            return label

# Test the TensorFlow Lite model
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    intent = predict_intent_tflite(user_input)
    responses = [response["responses"] for response in data["intents"] if response["tag"] == intent]
    if responses:
        print("Bot:", np.random.choice(responses[0]))
    else:
        print("Bot: Sorry, I didn't understand that.")
