# utils/recommender_engine.py

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------- Loaders ----------

@st.cache_data
def load_dialogue_embeddings():
    df = pd.read_parquet("data/dialogues_clean.parquet")
    df["embedding"] = df["embedding"].apply(np.array)  # Ensure proper type
    return df

@st.cache_data
def load_anime_metadata():
    return pd.read_parquet("data/anime.parquet")

@st.cache_resource
def get_transformer_model():
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")

# ---------- Recommender - From User Input ----------
def recommend_from_input(user_text, dialogue_df, anime_df, top_n=5):
    model = get_transformer_model()
    user_vector = model.encode([user_text])[0]

    # Filter rows with valid embeddings only
    valid_rows = dialogue_df[
        dialogue_df["embedding"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
    ].copy()

    # Stack embeddings and compute similarity
    embeddings_matrix = np.vstack(valid_rows["embedding"].values)
    scores = cosine_similarity([user_vector], embeddings_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_n]

    top_matches = valid_rows.iloc[top_indices].copy()
    top_matches["similarity"] = scores[top_indices]

    # 💡 Double-check this merge line
    if "anime" not in anime_df.columns and "name" in anime_df.columns:
        anime_df = anime_df.rename(columns={"name": "anime"})

    return top_matches.merge(anime_df, on="anime", how="left")
