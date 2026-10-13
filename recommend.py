"""Small command-line MoodTune demo: type a feeling, choose music language."""

import csv
import random
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SONGS_FILE = ROOT / "MoodTunes_Expanded_Final.csv"
USER_SONGS_FILE = Path(r"C:\Users\ahsan\Downloads\MoodTunes_Expanded_Final.csv")
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Map the classifier's six emotions to the five moods in the song catalogue.
MUSIC_MOOD = {
    "joy": "happy",
    "surprise": "energetic/motivated",
    "love": "romantic",
    "sadness": "sad",
    "anger": "calm/focussed",
    "fear": "calm/focussed",
}


def main():
    vectorizer_file = ARTIFACTS / "vectorizer.joblib"
    classifier_file = ARTIFACTS / "classifier.joblib"
    if not vectorizer_file.exists() or not classifier_file.exists():
        print("Train the model first: python train_model.py")
        return

    vectorizer = joblib.load(vectorizer_file)
    classifier = joblib.load(classifier_file)
    feeling = input("How are you feeling? Write in English: ").strip()
    if not feeling:
        print("Please enter a sentence about how you feel.")
        return

    label_id = int(classifier.predict(vectorizer.transform([feeling]))[0])
    emotion = LABELS[label_id]
    music_mood = MUSIC_MOOD[emotion]
    print(f"Predicted mood: {emotion} (music category: {music_mood})")

    language_choice = input("Song language (Malayalam / English / Hindi): ").strip().lower()
    language_names = {
        "malayalam": "Malayalam", "mal": "Malayalam",
        "english": "English", "eng": "English",
        "hindi": "Hindi", "hind": "Hindi",
    }
    if language_choice not in language_names:
        print("Choose Malayalam, English, or Hindi.")
        return
    language = language_names[language_choice].lower()

    songs_file = SONGS_FILE if SONGS_FILE.exists() else USER_SONGS_FILE
    if not songs_file.exists():
        print(f"Song catalogue not found. Expected it at: {USER_SONGS_FILE}")
        return

    with songs_file.open(encoding="utf-8-sig", newline="") as file:
        matching = [
            row for row in csv.DictReader(file)
            if row["language"].strip().lower() == language
            and row["mood"].strip().lower() == music_mood
        ]

    if len(matching) < 5:
        print(
            f"The song list currently has {len(matching)} matching songs. "
            "Add at least five rows for this language and mood in songs.csv."
        )
        return

    print(f"\nFive {language_names[language_choice]} songs for {music_mood} mood:")
    for number, song in enumerate(random.sample(matching, 5), start=1):
        print(f"{number}. {song['song_name']} — {song['artist_name']}")


if __name__ == "__main__":
    main()
