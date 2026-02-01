import pickle
from preprocess import clean_text

# STEP 3 STARTS HERE
# Load trained model and vectorizer (already trained on Kaggle)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# STEP 3: This function works for BOTH Kaggle news and NEW news
def predict_news(news_text):
    cleaned_text = clean_text(news_text)              # clean text
    vectorized_text = vectorizer.transform([cleaned_text])  # convert to numbers
    prediction = model.predict(vectorized_text)       # predict

    if prediction[0] == 1:
        return "Fake News ❌"
    else:
        return "Real News ✅"

# STEP 3 ENDS HERE


# OPTIONAL TEST (you can keep or remove this)
if __name__ == "__main__":
    test_news = """
    Breaking: Scientists announce a major breakthrough in renewable energy.
    """
    print(predict_news(test_news))
