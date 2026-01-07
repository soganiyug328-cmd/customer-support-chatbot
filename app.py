import json
import random
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
import nltk
import os

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)

texts = [preprocess(text) for text in texts]

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

def get_response(intent):
    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖")

st.title("🤖 Customer Support Chatbot")
st.write("Ask me about billing, refund, order status, or technical issues.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    processed = preprocess(user_input)
    vector = vectorizer.transform([processed])
    intent = model.predict(vector)[0]
    response = get_response(intent)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

