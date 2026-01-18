import streamlit as st
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(
    page_title="SonicMatcher Pro",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
    .prediction-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #1BFF1B;
        background-color: rgba(75, 255, 75, 0.05);
        margin-bottom: 1rem;
    }
    .top-genre {
        color: #3BFF3B;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .secondary-genre {
        font-size: 1rem;
        opacity: 0.8;
        margin: 5px 0;
    }
    .match-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    st_model = SentenceTransformer("models/st_transformer_model")
    bst = xgb.Booster()
    bst.load_model("models/genre_classifier.ubj")
    with open("models/metadata.json", "r") as f:
        meta = json.load(f)
    test_embs = np.load("models/test_embeddings.npy", mmap_mode='r')
    test_db = pd.read_feather("models/test_metadata.feather")
    return st_model, bst, meta, test_embs, test_db

if not os.path.exists("models/test_metadata.feather"):
    st.error("**File Error:** Assets not found. Please re-run your training script.")
    st.stop()

with st.spinner("Tuning up the engine..."):
    st_model, bst, meta, test_embs, test_db = load_assets()
    id_to_genre = meta["id_to_genre"]

st.title("🎵 SonicMatcher: Lyrics Analyzer")
st.write("Classify genres and find similar tracks across 11M+ songs.")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Input Lyrics")
    user_lyrics = st.text_area("Paste lyrics here:", height=300)
    analyze_btn = st.button("Run AI Analysis", use_container_width=True, type="primary")

with col_output:
    if analyze_btn and user_lyrics.strip():
        with st.spinner("Processing..."):
            # 1. Embeddings
            user_vec = st_model.encode([user_lyrics[:2000]])
            
            # 2. XGBoost Multi-Class Probabilities
            dmat = xgb.DMatrix(user_vec)
            prob_dist = bst.predict(dmat)[0] # Get the first row of probabilities
            
            # GET TOP 3 GENRE PREDICTIONS
            # argsort gives indices of sorted values; we take the last 3 for highest probs
            top_3_idx = np.argsort(prob_dist)[-3:][::-1]
            
            # 3. Similarity Search
            scores = cosine_similarity(user_vec, test_embs)[0]
            top_indices = np.argsort(scores)[-3:][::-1]
            matches = test_db.iloc[top_indices]
            match_scores = scores[top_indices]

        # DISPLAY TOP 3 GENRES
        st.subheader("Top Genre Predictions")
        for i, idx in enumerate(top_3_idx):
            g_name = id_to_genre[str(idx)]
            conf = prob_dist[idx]
            
            if i == 0: # Primary Prediction
                st.markdown(f"""
                <div class="prediction-card">
                    <p style="margin:0; font-size:0.8rem; opacity:0.7;">PRIMARY PREDICTION</p>
                    <p class="top-genre">{g_name}</p>
                    <p style="margin:0;">Confidence: {conf:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else: # Secondary Predictions
                st.write(f"**{i+1}. {g_name}** ({conf:.1%})")
                st.progress(float(conf))

        st.divider()

        # DISPLAY SIMILAR SONGS
        st.subheader("Similar Tracks You Might Like")
        for i in range(3):
            song_data = matches.iloc[i]
            # Use Song_Name if available, else fallback
            title = song_data.get('Song_Name', f"Match #{i+1}")
            st.markdown(f"""
            <div class="match-card">
                <div style="display:flex; justify-content:space-between;">
                    <strong>🎵 {title}</strong>
                    <span>{match_scores[i]:.1%} match</span>
                </div>
                <small>Original Genre: {song_data['Genre']}</small>
                <p style="font-style: italic; margin-top:5px; font-size:0.85rem;">
                    "{str(song_data['Lyrics'])[:120]}..."
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    elif analyze_btn:
        st.warning("Please enter some lyrics first!")
    else:
        st.info("Results will appear here after analysis.")

st.divider()
st.caption("Powered by Sentence Transformers & XGBoost | Optimized for 11M rows")