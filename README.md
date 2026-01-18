# 🎧 SonicMatcher Pro
### Real-Time Lyric Genre Classification & Similarity Search

SonicMatcher Pro is a high-performance NLP application designed to classify song lyrics and perform semantic similarity searches across a database of **11 million rows**. By leveraging **Sentence Transformers** for deep feature extraction and **XGBoost** for gradient-boosted classification, the app delivers lightning-fast music discovery and analysis.



##  Key Features
* **Deep Semantic Analysis:** Uses the `all-MiniLM-L6-v2` transformer to generate 384-dimensional dense vectors from raw lyrics.
* **Top-3 Prediction Engine:** Displays the primary predicted genre alongside the next two most likely candidates with real-time confidence scores.
* **11M Row Similarity Search:** Near-instant discovery of similar songs using cosine similarity.
* **Memory-Optimized Engine:** * **NumPy Mem-Map:** Accesses 16GB+ of embeddings without high RAM overhead.
    * **Feather Storage:** Optimized IPC format for rapid metadata retrieval of song titles and lyrics.
* **Adaptive UI:** Responsive Streamlit interface with full Light/Dark mode support.

## Tech Stack
* **Embeddings:** `sentence-transformers` (Hugging Face)
* **Classifier:** `xgboost` (Booster API)
* **Frontend:** `streamlit`
* **Data Format:** `pyarrow` (Feather) & `numpy` (.npy)
* **Similarity Engine:** `scikit-learn` (Cosine Similarity)

---

## Project Structure
```text
project/
│── models/
│   ├── st_transformer_model/    # Local Sentence Transformer weights
│   ├── genre_classifier.ubj     # Trained XGBoost Booster (UBJSON)
│   ├── metadata.json            # Genre ID to Name mappings
│   ├── test_embeddings.npy      # 11M vectors (Memory-mapped)
│   └── test_metadata.feather    # Song titles/lyrics (Feather format)
│── train_script.py              # Streaming training & saving script
│── app.py                       # Streamlit application script
└── requirements.txt             # Project dependencies
