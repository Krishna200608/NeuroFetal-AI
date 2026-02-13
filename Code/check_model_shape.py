import tensorflow as tf
import sys
import io

# Force UTF-8 for Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_path = "models/enhanced_model_fold_1.keras"

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    for i, inp in enumerate(model.inputs):
        print(f"Input {i}: {inp.shape} name={inp.name}")
except Exception as e:
    print(f"Error loading model: {e}")
