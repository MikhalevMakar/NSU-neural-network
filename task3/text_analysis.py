import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

texts = [
    "Этот фильм был очень хороший и интересный.",
    "Сегодняшняя погода ужасная, совсем не радует.",
    "Я вчера побывал на концерте и он был прекрасным.",
    "У меня был ужасный день, ничего не получалось.",
    "Этот ресторан предоставил отличное обслуживание и вкусную еду.",
    "Я разочарован качеством этого товара, не рекомендую его.",
    "Великолепный выходной день, все прошло замечательно!",
    "Этот сериал скучный и предсказуемый, не стоит его смотреть."
]
sentiments = [
    "Положительная",
    "Негативная",
    "Положительная",
    "Негативная",
    "Положительная",
    "Негативная",
    "Положительная",
    "Негативная"
]


def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)


preprocessed_texts = [preprocess_text(text) for text in texts]

tfidf_vectorizer = TfidfVectorizer()

tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_texts)

classifier = LogisticRegression()
classifier.fit(tfidf_features, sentiments)


def analyze_sentiment(text):
    preprocessed_text = preprocess_text(text)
    tfidf_features_text = tfidf_vectorizer.transform([preprocessed_text])
    sentiment = classifier.predict(tfidf_features_text)
    return sentiment[0]


new_text = "Этот книга была очень интересной и вдохновляющей."
result = analyze_sentiment(new_text)
print("Тональность текста:", result)
