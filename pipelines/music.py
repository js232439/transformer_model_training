from transformers import pipeline

# Initialize zero-shot-classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# List of songs (song titles or descriptions)
songs = [
    "Lose Yourself by Eminem",
    "Bohemian Rhapsody by Queen",
    "Shape of You by Ed Sheeran",
    "Billie Jean by Michael Jackson",
    "Blinding Lights by The Weeknd",
    "Always by Daniel Caesar"
]

# List of possible genres
genres = ["hip hop", "rock", "pop", "classical", "jazz", "electronic", "R&B"]

# Classify each song by genre
for song in songs:
    result = classifier(song, genres)
    print(f"Song: {song}")
    print(f"Predicted Genre: {result['labels'][0]} with score {result['scores'][0]:.2f}")
    print()
