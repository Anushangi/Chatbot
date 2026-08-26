import json
import nltk
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
import string

def stem(word):
    return stemmer.stem(word.lower())

# Download NLTK tokenizer (first time only)
nltk.download('punkt')
nltk.download('punkt_tab')

# Load intents.json
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

print(data)
# Tokenize all patterns

words = []
tags = []
xy = []

for intent in data["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)

        words.extend(tokens)
        xy.append((tokens, tag))

    if tag not in tags:
        tags.append(tag)

print("\nWords:")
print(words)

print("\nTags:")
print(tags)

print("\nTraining Data:")
print(xy)
print("\nStemmed Words:")

stemmed_words = [
    stem(word)
    for word in words
    if word not in string.punctuation
]

print(stemmed_words)
# Remove duplicate words and sort them
words = sorted(set(stemmed_words))
tags = sorted(set(tags))

print("\nVocabulary:")
print(words)

print("\nTotal Vocabulary Size:")
print(len(words))
# Create training data

training = []

for (pattern_sentence, tag) in xy:

    # Stem each word in the sentence
    pattern_words = [stem(word) for word in pattern_sentence]

    # Create Bag of Words
    bag = []

    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    training.append((bag, tag))

print("\nBag of Words:")
for item in training:
    print(item)
    # Prepare training data

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

print("\nSentences:")
print(sentences)

print("\nLabels:")
print(labels)
# Convert text into numerical features

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(sentences)

# Train model

model = LinearSVC()
model.fit(X, labels)

print("\nModel trained successfully!")

# Save model and vectorizer

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("model.pkl saved")
print("vectorizer.pkl saved")