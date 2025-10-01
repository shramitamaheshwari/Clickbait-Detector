import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('clickbait_data.csv')
print(df.head())
df = df.dropna()
df['headline'] = df['headline'].str.lower()


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['headline'])
y = df['clickbait']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def predict_clickbait(text):
    vec = vectorizer.transform([text])
    return "Clickbait" if model.predict(vec)[0] == 1 else "Not Clickbait"

print(predict_clickbait("You won't believe what happened next!"))

import joblib

joblib.dump(model, 'clickbait_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
