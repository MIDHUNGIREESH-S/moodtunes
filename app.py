"""Shareable Streamlit MoodTune app using the locally trained model."""

import csv
import html
import random
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SONGS_FILE = ROOT / "MoodTunes_Expanded_Final.csv"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MUSIC_MOOD = {
    "joy": "happy",
    "surprise": "energetic/motivated",
    "love": "romantic",
    "sadness": "sad",
    "anger": "calm/focussed",
    "fear": "calm/focussed",
}
LANGUAGES = {
    "Malayalam": "Malayalam",
    "English": "English",
    "Hindi": "Hindi",
}
MOOD_COLORS = {
    "sad": "#3b82f6",
    "happy": "#facc15",
    "angry": "#ef4444",
    "calm": "#22c55e",
    "focused": "#a855f7",
}

st.set_page_config(
    page_title="Mood Tunes - A Music Recommender",
    page_icon="🎵",
    layout="centered",
)


@st.cache_resource
def load_model():
    vectorizer_file = ARTIFACTS / "vectorizer.joblib"
    classifier_file = ARTIFACTS / "classifier.joblib"
    if not vectorizer_file.exists() or not classifier_file.exists():
        raise FileNotFoundError(
            "Model files are missing. Include the artifacts folder when publishing."
        )
    return joblib.load(vectorizer_file), joblib.load(classifier_file)


@st.cache_data
def load_songs():
    if not SONGS_FILE.exists():
        raise FileNotFoundError(f"Song catalogue not found: {SONGS_FILE.name}")
    with SONGS_FILE.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def get_recommendations(user_text: str, language: str):
    vectorizer, classifier = load_model()
    all_songs = load_songs()

    label_id = int(classifier.predict(vectorizer.transform([user_text]))[0])
    emotion = LABELS[label_id]
    music_mood = MUSIC_MOOD[emotion]

    matching = [
        song for song in all_songs
        if song["language"].strip().casefold() == language.casefold()
        and song["mood"].strip().casefold() == music_mood.casefold()
    ]
    if len(matching) < 5:
        raise ValueError(
            f"Only {len(matching)} songs are available for {language} / {music_mood}."
        )

    songs = random.sample(matching, 5)
    return emotion, music_mood, language, songs


st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f19; color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #111827; border-right: 1px solid #1f2937;
    }
    h1 {
        background: linear-gradient(90deg, #ff4b4b, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    textarea {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        caret-color: #000000 !important;
    }
    .song-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff4b4b, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎧 Mood Tunes")
    st.markdown("---")
    st.header("⚙️ Controls")
    language_filter = st.selectbox(
        "🌍 Select your song language",
        list(LANGUAGES),
    )

st.title("🎵 Mood Tunes")
st.markdown("##### *Transforming your emotions into the ultimate soundtrack.*")
st.markdown("### 🧠 How is your mind feeling right now?")
user_input = st.text_area(
    "Describe your thoughts or current situation in English:",
    placeholder="e.g., I'm feeling sad today...",
    height=100,
)

if st.button("✨ Analyze Mood & Recommend Songs"):
    if not user_input.strip():
        st.warning("⚠️ Please type something about how you are feeling first!")
    else:
        with st.spinner("Analyzing your mood..."):
            try:
                emotion, music_mood, language, songs = get_recommendations(
                    user_input.strip(), language_filter
                )
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
                songs = []

        if songs:
            color = next(
                (value for key, value in MOOD_COLORS.items() if key in music_mood),
                "#ff4b4b",
            )
            st.success(
                f"🎯 **Detected emotion:** `{emotion}`  ·  **Music mood:** `{music_mood}`"
            )
            st.markdown(f"### 🎧 Five {language} tracks for {music_mood}")

            for song in songs:
                title = html.escape(song["song_name"])
                artist = html.escape(song["artist_name"])
                tag = html.escape(music_mood)
                query = quote_plus(f"{song['song_name']} {song['artist_name']}")
                youtube_link = f"https://www.youtube.com/results?search_query={query}"
                st.markdown(
                    f"""
                    <div class="song-card" style="border-left: 5px solid {color};">
                        <h4>🎶 {title}
                            <span style="font-size: 12px; color: #9ca3af;
                                         float: right;">[{tag}]</span>
                        </h4>
                        <p style="margin: 0; color: #d1d5db;">
                            Artist: <b>{artist}</b> |
                            Language: <b>{language}</b>
                        </p>
                        <a href="{youtube_link}" target="_blank"
                           style="color: #60a5fa; font-weight: bold;
                                  text-decoration: none; display: inline-block;
                                  margin-top: 5px;">
                            ▶ Watch / Listen on YouTube
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.toast("Done! Enjoy your music 🎶")
