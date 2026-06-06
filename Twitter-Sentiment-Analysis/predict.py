import argparse
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def predict(text: str) -> str:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files not found. Run 'python main.py' first.")

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    confidence = max(probabilities)
    label = "positive" if prediction == 1 else "negative"
    return f"{label} (confidence: {confidence:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for a tweet")
    parser.add_argument(
        "text",
        nargs="?",
        default="I love this amazing product and it works great!",
        help="Text to classify",
    )
    args = parser.parse_args()

    print("Prediction:", predict(args.text))


if __name__ == "__main__":
    main()
