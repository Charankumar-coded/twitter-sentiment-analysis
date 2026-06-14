import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os
from pathlib import Path

# Get current directory
current_dir = Path(__file__).resolve().parent
data_file = current_dir / "twitter_sentiment.csv"
model_file = current_dir / "sentiment_model.pkl"

print("Loading Twitter sentiment data...")
df = pd.read_csv(data_file)

print(f"Data shape: {df.shape}")
print(f"Sample data:\n{df.head()}\n")

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
print("Training sentiment classifier...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('nb', MultinomialNB())
])

# Train the model
X = df['text']
y = df['label']
model.fit(X, y)

# Save the model
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_file}")

# Print some sample predictions
print("\nSample predictions:")
sample_texts = [
    "I love this!",
    "This is terrible",
    "Amazing product",
    "Worst experience ever"
]

for text in sample_texts:
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment} (confidence: {max(probability):.2%})\n")
