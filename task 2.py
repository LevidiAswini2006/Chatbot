import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Only needed once
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------
# STEP 1: Define FAQ dataset
# ------------------------------
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "Do you ship internationally?": "Yes, we ship to most countries worldwide.",
    "How do I reset my password?": "Click on 'Forgot Password' at login and follow the steps.",
    "What payment methods are accepted?": "We accept credit/debit cards, UPI, and PayPal."
}

# ------------------------------
# STEP 2: Preprocess text
# ------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

questions = list(faq_data.keys())
answers = list(faq_data.values())

preprocessed_questions = [preprocess(q) for q in questions]

# ------------------------------
# STEP 3: Vectorization and Matching
# ------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = np.argmax(similarity)
    if similarity[0][best_match_index] < 0.2:
        return "Sorry, I couldn't understand your question."
    return answers[best_match_index]

# ------------------------------
# STEP 4: Chat Loop
# ------------------------------
print("🤖 FAQ Chatbot is ready! (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
