import pickle
import sys
from pathlib import Path

# Get current directory
current_dir = Path(__file__).resolve().parent
model_file = current_dir / "sentiment_model.pkl"

# Load the trained model
if not model_file.exists():
    print("Error: Model not found. Please run main.py first to train the model.")
    sys.exit(1)

with open(model_file, 'rb') as f:
    model = pickle.load(f)

print("Sentiment Analysis Prediction")
print("=" * 50)

# If arguments provided, predict on those
if len(sys.argv) > 1:
    text = ' '.join(sys.argv[1:])
else:
    text = input("Enter text to analyze: ")

# Make prediction
prediction = model.predict([text])[0]
probability = model.predict_proba([text])[0]

sentiment = "POSITIVE ✓" if prediction == 1 else "NEGATIVE ✗"
confidence = max(probability)

print(f"\nText: '{text}'")
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
print("=" * 50)
