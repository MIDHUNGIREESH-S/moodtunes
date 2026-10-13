"""Local API for connecting MoodTune's Python model to a frontend."""

import csv
import random
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SONGS_FILE = ROOT / "MoodTunes_Expanded_Final.csv"
USER_SONGS_FILE = Path(r"C:\Users\ahsan\Downloads\MoodTunes_Expanded_Final.csv")
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
    "malayalam": "Malayalam", "mal": "Malayalam",
    "english": "English", "eng": "English",
    "hindi": "Hindi", "hind": "Hindi",
}


class RecommendationRequest(BaseModel):
    text: str = Field(min_length=1, description="How the user feels, written in English")
    language: str = Field(description="Malayalam, English, or Hindi")


def load_songs():
    path = SONGS_FILE if SONGS_FILE.exists() else USER_SONGS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Song catalogue not found: {USER_SONGS_FILE}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


vectorizer_path = ARTIFACTS / "vectorizer.joblib"
classifier_path = ARTIFACTS / "classifier.joblib"
if not vectorizer_path.exists() or not classifier_path.exists():
    raise RuntimeError("Train the model first by running: python train_model.py")

vectorizer = joblib.load(vectorizer_path)
classifier = joblib.load(classifier_path)
songs = load_songs()

app = FastAPI(title="MoodTune API", version="1.0.0")

# Development setting for local frontend testing. Restrict this to the deployed
# frontend's origin before putting the API on a public server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "song_count": len(songs)}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Please enter how you feel in English.")

    language_choice = request.language.strip().lower()
    language = LANGUAGES.get(language_choice)
    if language is None:
        raise HTTPException(
            status_code=422,
            detail="Choose Malayalam, English, or Hindi.",
        )

    label_id = int(classifier.predict(vectorizer.transform([text]))[0])
    emotion = LABELS[label_id]
    music_mood = MUSIC_MOOD[emotion]
    matching = [
        song for song in songs
        if song["language"].strip().casefold() == language.casefold()
        and song["mood"].strip().casefold() == music_mood.casefold()
    ]
    if len(matching) < 5:
        raise HTTPException(
            status_code=503,
            detail=f"Only {len(matching)} matching songs are available; five are needed.",
        )

    selected = random.sample(matching, 5)
    return {
        "input_text": text,
        "language": language,
        "predicted_emotion": emotion,
        "music_mood": music_mood,
        "songs": [
            {"title": song["song_name"], "artist": song["artist_name"]}
            for song in selected
        ],
    }
