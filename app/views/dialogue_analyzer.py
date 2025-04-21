import streamlit as st
import pandas as pd
import plotly.express as px

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.subtitle_parser import parse_srt_file, parse_ass_file
from utils.sentiment_analysis import filter_by_mood, count_moods, classify_emotions_transformer



def show():
    st.title("💬 Dialogue Analyzer")
    st.subheader("Upload a subtitle file and explore its emotional mood spectrum.")

    uploaded_file = st.file_uploader("📤 Upload subtitle (.srt or .ass)", type=["srt", "ass"])
    if not uploaded_file:
        st.info("Please upload a subtitle file to begin.")
        return

    # File parsing
    if uploaded_file.name.endswith(".srt"):
        dialogues = parse_srt_file(uploaded_file)
    else:
        dialogues = parse_ass_file(uploaded_file)

    dialogues = [d.strip() for d in dialogues if d.strip()]
    st.success(f"✅ Parsed {len(dialogues)} dialogue lines.")

    # Model selection
    model_choice = st.radio("Choose Sentiment Model:", ["VADER", "Transformer 🤗"])
    sid = SentimentIntensityAnalyzer()

    if model_choice == "Transformer 🤗":
        MAX_LEN = 512

        # Clean + truncate too-long dialogues
        clean_dialogues = [
            d for d in dialogues 
            if len(d.strip()) > 5 and len(d.strip()) < MAX_LEN
        ]
        
        labels = classify_emotions_transformer(clean_dialogues)

        st.write(f"Number of clean dialogues: {len(clean_dialogues)}")
        st.write(f"Number of labels returned: {len(labels)}")

        if len(clean_dialogues) != len(labels):
            st.error("Mismatch in number of dialogues and labels.")
            st.stop()

        df = pd.DataFrame({"dialogue": clean_dialogues, "label": labels})
        mood_order = df["label"].value_counts().index.tolist()
        mood_counts = df["label"].value_counts().to_dict()

    else:
        mood_counts = count_moods(dialogues, sid)
        df = pd.DataFrame({
            "dialogue": dialogues,
            "label": [get_vader_label(d, sid) for d in dialogues]
        })
        mood_order = [
            "Joyful 😄", "Romantic 💖", "Hopeful 🌈",
            "Melancholic 😔", "Tense 😬", "Angry 😠", "Empty 😶"
        ]

    # Mood chart
    mood_df = pd.DataFrame({
        "Mood": list(mood_counts.keys()),
        "Count": list(mood_counts.values())
    }).set_index("Mood").reindex(mood_order).dropna().reset_index()

    fig = px.bar(
        mood_df, y="Mood", x="Count",
        orientation="h", color="Mood",
        color_discrete_sequence=[
            "#FFD1DC", "#FFB7CE", "#B5EAD7", "#AEC6CF",
            "#CBAACB", "#FF6961", "#E0E0E0"
        ],
        title="🌈 Emotional Mood Spectrum"
    )
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor="#fff9f9",
        paper_bgcolor="#fff9f9",
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis_title="", yaxis_title="", showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mood filter
    selected_mood = st.selectbox("🎭 Filter by mood:", ["All"] + mood_order)

    if selected_mood == "All":
        st.markdown("### ✨ Showing sample dialogues")
        st.code("\n".join(df["dialogue"].head(10)))
    else:
        st.markdown(f"### 🎯 Showing **{selected_mood}** dialogues")
        filtered = df[df["label"] == selected_mood]["dialogue"].tolist()
        st.code("\n".join(filtered[:10]))


def get_vader_label(text, sid):
    score = sid.polarity_scores(text)["compound"]
    if score >= 0.5:
        return "Joyful 😄"
    elif 0.3 <= score <= 0.6:
        return "Romantic 💖"
    elif 0.2 <= score <= 0.5:
        return "Hopeful 🌈"
    elif -0.6 <= score <= -0.2:
        return "Melancholic 😔"
    elif -1.0 <= score <= -0.7:
        return "Tense 😬"
    elif score <= -0.6:
        return "Angry 😠"
    elif -0.1 <= score <= 0.1:
        return "Empty 😶"
    else:
        return "Neutral"
