from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import streamlit as st

def analyze_sentiment(dialogues):
    sid = SentimentIntensityAnalyzer()
    scores = {"positive": 0, "neutral": 0, "negative": 0}

    for d in dialogues:
        score = sid.polarity_scores(d)["compound"]
        if score >= 0.05:
            scores["positive"] += 1
        elif score <= -0.05:
            scores["negative"] += 1
        else:
            scores["neutral"] += 1

    total = len(dialogues)
    return {k: f"{(v/total)*100:.2f}%" for k, v in scores.items()}


def filter_by_mood(dialogues, mood, sid):
    mood_scores = {
        "Joyful 😄": lambda s: s >= 0.5,
        "Melancholic 😔": lambda s: -0.6 <= s <= -0.2,
        "Tense 😬": lambda s: -1.0 <= s <= -0.7,
        "Romantic 💖": lambda s: 0.3 <= s <= 0.6,
        "Angry 😠": lambda s: s <= -0.6,
        "Hopeful 🌈": lambda s: 0.2 <= s <= 0.5,
        "Empty 😶": lambda s: -0.1 <= s <= 0.1,
    }
    

    if mood == "All":
        return dialogues

    filter_func = mood_scores.get(mood, lambda s: True)
    return [d for d in dialogues if filter_func(sid.polarity_scores(d)["compound"])]


def count_moods(dialogues, sid):
    mood_labels = {
        "Joyful 😄": lambda s: s >= 0.5,
        "Melancholic 😔": lambda s: -0.6 <= s <= -0.2,
        "Tense 😬": lambda s: -1.0 <= s <= -0.7,
        "Romantic 💖": lambda s: 0.3 <= s <= 0.6,
        "Angry 😠": lambda s: s <= -0.6,
        "Hopeful 🌈": lambda s: 0.2 <= s <= 0.5,
        "Empty 😶": lambda s: -0.1 <= s <= 0.1,
    }

    counts = {mood: 0 for mood in mood_labels}
    for d in dialogues:
        compound = sid.polarity_scores(d)["compound"]
        for mood, condition in mood_labels.items():
            if condition(compound):
                counts[mood] += 1
                break

    return counts


import streamlit as st
from transformers import pipeline

@st.cache_data(show_spinner="Analyzing with Transformer... 🔍")
def classify_emotions_transformer(dialogues):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    results = classifier(dialogues)
    return [res[0]["label"] for res in results]
