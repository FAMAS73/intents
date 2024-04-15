import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("intent_model.h5")

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Set quantization parameters for further model size reduction
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("intent_model.tflite", "wb") as f:
  f.write(tflite_model)
