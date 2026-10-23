"""16-dimensional demo dataset and a toy keyword encoder.

The 16 dimensions are split into four topic regions:
    dims  0-3   tech
    dims  4-7   space
    dims  8-11  music
    dims 12-15  sports

A vector's value in a region says how strongly the item belongs to that
topic. Real embedding models learn dimensions like this on their own;
here they are designed by hand so the behaviour is easy to inspect.
Some items deliberately mix topics (the ISS is space + tech), so they sit
between clusters on the scatter plot.
"""

import numpy as np

DIMS = 16
CATEGORIES = ("tech", "space", "music", "sports")
_REGION = {c: i * 4 for i, c in enumerate(CATEGORIES)}
_BACKGROUND = 0.06

# (text, primary category, topic weights)
DEMO_ITEMS: list[tuple[str, str, dict[str, float]]] = [
    ("Python: readable language with a huge library ecosystem", "tech", {"tech": 0.92}),
    ("Git: version control with branches, commits and merges", "tech", {"tech": 0.88}),
    ("Docker: containers that package an app with its dependencies", "tech", {"tech": 0.90}),
    ("REST APIs: HTTP endpoints exchanging JSON", "tech", {"tech": 0.86}),
    ("Neural networks: layers of weights trained by backpropagation", "tech", {"tech": 0.91}),
    ("Black holes: gravity so strong that light cannot escape", "space", {"space": 0.93}),
    ("Mars: the red planet with the tallest volcano in the solar system", "space", {"space": 0.89}),
    ("Apollo 11: the first crewed Moon landing in 1969", "space", {"space": 0.87}),
    ("Exoplanets: worlds orbiting stars beyond the Sun", "space", {"space": 0.90}),
    ("The ISS: a science laboratory orbiting Earth", "space", {"space": 0.80, "tech": 0.35}),
    ("Guitar: chords and riffs played on six strings", "music", {"music": 0.91}),
    ("Piano: 88 keys spanning bass to treble", "music", {"music": 0.88}),
    ("Hip-hop: beats, rhythm and rap verses", "music", {"music": 0.90}),
    ("Classical music: symphonies by Mozart and Beethoven", "music", {"music": 0.87}),
    ("Music production: mixing and mastering tracks in a DAW", "music", {"music": 0.80, "tech": 0.38}),
    ("Cricket: bat, ball, wickets and overs", "sports", {"sports": 0.92}),
    ("Football: dribbling, passing and scoring goals", "sports", {"sports": 0.90}),
    ("Badminton: fast rallies with a shuttlecock", "sports", {"sports": 0.88}),
    ("Marathon: running 42 kilometres of endurance", "sports", {"sports": 0.86}),
    ("Chess: openings, tactics and checkmates", "sports", {"sports": 0.78, "tech": 0.30}),
]

KEYWORDS: dict[str, list[str]] = {
    "tech": ["tech", "code", "coding", "program", "python", "software", "computer", "git",
             "docker", "api", "web", "server", "data", "algorithm", "ai", "machine", "neural",
             "network", "app", "database", "cloud", "developer", "linux", "javascript"],
    "space": ["space", "planet", "star", "galaxy", "nasa", "rocket", "moon", "mars", "orbit",
              "astronaut", "universe", "black", "hole", "solar", "sun", "telescope", "comet",
              "satellite", "apollo", "iss", "exoplanet", "cosmos", "alien"],
    "music": ["music", "song", "guitar", "piano", "beat", "rap", "hip", "hop", "melody",
              "chord", "band", "concert", "album", "sing", "singer", "drum", "violin",
              "classical", "mozart", "beethoven", "rhythm", "lyrics", "track", "dj"],
    "sports": ["sport", "cricket", "football", "soccer", "badminton", "tennis", "chess",
               "marathon", "run", "running", "goal", "match", "team", "player", "olympic",
               "wicket", "game", "score", "athlete", "tournament", "fitness", "gym", "race"],
}


def make_vector(weights: dict[str, float], seed: int) -> np.ndarray:
    """Build a 16D vector from topic weights, with a little deterministic
    variation so items in the same topic are close but not identical."""
    rng = np.random.default_rng(seed)
    v = np.full(DIMS, _BACKGROUND, dtype=np.float32)
    v += rng.uniform(-0.03, 0.03, DIMS).astype(np.float32)
    for category, w in weights.items():
        start = _REGION[category]
        shape = rng.uniform(0.85, 1.0, 4)  # unique profile inside the region
        v[start:start + 4] = (w * shape).astype(np.float32)
    return np.clip(v, 0.01, 1.0)


def demo_vectors() -> list[tuple[str, str, np.ndarray]]:
    return [(text, cat, make_vector(w, seed=i)) for i, (text, cat, w) in enumerate(DEMO_ITEMS)]


def _matches(word: str, keyword: str) -> bool:
    if word == keyword:
        return True
    # Prefix matching for word forms ("planets" -> "planet"), but only for
    # longer stems so "a" or "go" don't match everything.
    return len(word) >= 4 and len(keyword) >= 4 and (word.startswith(keyword) or keyword.startswith(word))


def baseline_vector() -> np.ndarray:
    """The vector for text with no known keywords: no topic, background level only."""
    return np.full(DIMS, _BACKGROUND, dtype=np.float32)


def encode_text(text: str) -> np.ndarray | None:
    """Toy encoder: map free text to a 16D demo vector by keyword matching.

    This stands in for an embedding model on the demo dataset only.
    Returns None if no known keyword appears in the text.
    """
    words = [w.strip(".,!?;:'\"()[]").lower() for w in text.split()]
    scores = {c: 0 for c in CATEGORIES}
    for word in filter(None, words):
        for category, keywords in KEYWORDS.items():
            if any(_matches(word, kw) for kw in keywords):
                scores[category] += 1
    best = max(scores.values())
    if best == 0:
        return None
    weights = {c: 0.9 * s / best for c, s in scores.items() if s > 0}
    v = np.full(DIMS, _BACKGROUND, dtype=np.float32)
    for category, w in weights.items():
        start = _REGION[category]
        v[start:start + 4] = w
    return v
