from flask import Flask, render_template, request, jsonify
import json
import pickle
import random

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot():

    message = request.form["msg"].strip().lower()

    # Convert message into vector
    X = vectorizer.transform([message])

    # Prediction
    prediction = model.predict(X)[0]
    scores = model.decision_function(X)

    print("Message :", message)
    print("Prediction :", prediction)
    print("Score :", max(scores[0]))

    # Unknown question detection
    if max(scores[0]) < 0.4:
      return jsonify({
        "reply": "I'm sorry, I don't have information about that. Please ask another college-related question."
    })

    # Return matching response
    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return jsonify({
                "reply": random.choice(intent["responses"])
            })

    return jsonify({
        "reply": "I'm sorry, I don't have information about that.ask another question"
    })


if __name__ == "__main__":
    app.run(debug=True)