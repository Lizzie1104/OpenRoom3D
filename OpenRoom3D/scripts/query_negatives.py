"""Default negative prompts and scoring rule for text queries.

The scene categories were written by hand for the desk02 tabletop. On that scene
they raised laptop average precision from 0.415 to 0.491 and cut look-alike
`television` responses from 195 to 16 points (outputs/retrain_semantic_2026-11-02/negatives).
Replace them with categories visible in a new scene, and report them as manual input.
"""

SCORE_MODES = ("softmax", "pairwise-min")
DEFAULT_SCORE_MODE = "pairwise-min"
# LERF's canonical phrases.
CANONICAL_NEGATIVES = "object,things,stuff,texture"
SCENE_NEGATIVES = "table,desk,chair,floor,wall,computer mouse,water bottle,cable,paper,mat"
DEFAULT_NEGATIVES = CANONICAL_NEGATIVES + "," + SCENE_NEGATIVES


def without_query(negatives, query):
    """Drop negatives sharing a word with a positive query.

    A query cannot be its own negative: "mouse" against "computer mouse" would
    cap every score near 0.5.
    """
    words = {word for text in query.lower().split(",") for word in text.split()}
    kept = [text.strip() for text in negatives.split(",")
            if text.strip() and not words & set(text.lower().split())]
    if not kept:
        raise ValueError("every negative matches the query; supply other negatives")
    return ",".join(kept)
