# views/recommender.py

import streamlit as st
from utils.recommender_engine import (
    load_dialogue_embeddings,
    load_anime_metadata,
    get_transformer_model,
    recommend_from_input
)

# Load required data
anime_df = load_anime_metadata()
dialogue_df = load_dialogue_embeddings()


def show():
    st.title("🎯 Anime Recommender")
    st.subheader("Find anime with matching dialogue vibes ✨")

    st.markdown("### 💬 Enter Your Dialogue")
    user_input = st.text_area("Paste a short anime-like dialogue:", height=150)

    if st.button("🔍 Find Matching Anime") and user_input.strip():
        with st.spinner("Matching your vibes... please wait ⏳"):
            recommendations = recommend_from_input(user_input, dialogue_df, anime_df)

        recommendations = recommend_from_input(user_input, dialogue_df, anime_df)

        if not recommendations.empty:
            st.success("Top 5 Matches:")
            for _, row in recommendations.iterrows():
                st.markdown(f"**{row['anime']}**")
                st.caption(f"Similarity: {row['similarity']:.2f}")
                st.markdown("---")
        else:
            st.warning("No close matches found. Try a different dialogue ✨")
