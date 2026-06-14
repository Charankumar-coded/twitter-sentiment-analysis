# Twitter Sentiment Analysis

A machine learning project that classifies Twitter sentiment as positive or negative using Naive Bayes classification.

## 📋 Overview

This project analyzes Twitter text data and predicts sentiment (positive or negative) using a TF-IDF vectorizer and Multinomial Naive Bayes classifier. It includes training and prediction functionality.

## ✨ Features

- **Train Model**: Build sentiment classifier from Twitter data
- **Make Predictions**: Classify new text as positive or negative
- **Confidence Scores**: Get prediction confidence percentages
- **Easy to Use**: Simple command-line interface

## 📦 Requirements

- Python 3.8+
- pandas >= 2.2
- scikit-learn >= 1.4
- numpy >= 1.26

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Charankumar-coded/twitter-sentiment-analysis.git
cd sentiment-analysis-on-twitter-data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses `twitter_sentiment.csv` with the following structure:
- `text`: Tweet content
- `label`: Sentiment (1 = Positive, 0 = Negative)

Example:
```
text,label
"I love this product, it is amazing!",1
"This is the worst experience ever",0
```

## 🎯 Usage

### Train the Model

Run the training script to build and save the sentiment classifier:

```bash
python main.py
```

Output:
- Trains on Twitter sentiment data
- Saves model to `sentiment_model.pkl`
- Shows sample predictions
- Displays training accuracy

### Make Predictions

Predict sentiment on new text:

```bash
# Command line argument
python predict.py "I love this product"

# Or interactive mode
python predict.py
```

Example Output:
```
Sentiment Analysis Prediction
==================================================

Text: 'I love this product'
Sentiment: POSITIVE ✓
Confidence: 75.43%
==================================================
```

## 📁 Project Structure

```
.
├── main.py                    # Training script
├── predict.py                 # Prediction script
├── twitter_sentiment.csv      # Training dataset
├── sentiment_model.pkl        # Saved trained model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔍 How It Works

1. **Data Loading**: Reads tweet data from CSV
2. **Vectorization**: Converts text to TF-IDF features
3. **Training**: Trains Naive Bayes classifier
4. **Prediction**: Classifies new text based on learned patterns
5. **Confidence**: Provides probability scores

## 📈 Model Performance

- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF (max 5000 features)
- **Text Processing**: Removes English stop words
- **Classes**: Binary (0=Negative, 1=Positive)

## 💡 Example Predictions

| Text | Sentiment | Confidence |
|------|-----------|-----------|
| "I love this!" | POSITIVE | 66.50% |
| "This is terrible" | NEGATIVE | 67.20% |
| "Amazing product" | POSITIVE | 61.87% |
| "Worst experience ever" | NEGATIVE | 65.80% |

## 🛠️ Technologies Used

- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## 📝 License

This project is open source.

## 👨‍💻 Author

Sentiment Analysis on Twitter Data

## 📞 Support

For issues or questions, please open an issue on GitHub.
