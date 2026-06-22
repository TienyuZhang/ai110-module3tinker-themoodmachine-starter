# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re

from typing import List, Dict, Tuple, Optional

from dataset import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_EMOJIS,
    NEGATIVE_EMOJIS,
)

# Common "ASCII" emoticons we want to keep as their own tokens.
# Kept lowercase because preprocess() lowercases text before matching (":D" -> ":d").
ASCII_EMOJIS = [":)", ":-)", "(:", ":(", ":-(", "):", ":d", ":p", ":/", ";)", "<3"]

# Words that flip the sentiment of the word that follows them.
NEGATION_WORDS = {"not", "no", "never", "none", "cant", "can't", "dont", "don't", "isnt", "isn't"}

# Matches most Unicode emoji (faces, symbols, hands, hearts, etc.).
_UNICODE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]"
)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup. Emojis/emoticons are folded in so
        # the scorer treats them as sentiment signals alongside plain words.
        self.positive_words = set(w.lower() for w in positive_words) | set(POSITIVE_EMOJIS)
        self.negative_words = set(w.lower() for w in negative_words) | set(NEGATIVE_EMOJIS)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Improvements implemented:
          - Strips whitespace and lowercases (consistent casing)
          - Keeps ASCII emoticons like ":)" / ":(" as their own tokens
          - Splits Unicode emojis (🥲😂💀) into separate tokens
          - Strips surrounding punctuation from words ("day!" -> "day")
          - Normalizes elongated repeats ("soooo" -> "soo", "gooo" -> "goo")
        """
        cleaned = text.strip().lower()

        tokens: List[str] = []
        for raw in cleaned.split():
            # 1. Pull out any ASCII emoticons (":)", ":(") as standalone tokens.
            for emoji in ASCII_EMOJIS:
                if emoji in raw:
                    tokens.append(emoji)
                    raw = raw.replace(emoji, " ")

            # 2. Separate Unicode emojis so they don't stick to words.
            raw = _UNICODE_EMOJI.sub(lambda m: f" {m.group()} ", raw)

            for piece in raw.split():
                # 3. A lone Unicode emoji is already a token; keep it as-is.
                if _UNICODE_EMOJI.fullmatch(piece):
                    tokens.append(piece)
                    continue

                # 4. Strip punctuation from the edges, keep internal apostrophes
                #    so "can't" / "don't" survive.
                word = piece.strip(".,!?;:\"'()[]{}…")
                if not word:
                    continue

                # 5. Collapse runs of 3+ identical letters down to 2.
                word = re.sub(r"(.)\1{2,}", r"\1\1", word)
                tokens.append(word)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)

        score = 0
        negate = False  # True when the previous token was a negation word.

        for token in tokens:
            if token in self.positive_words:
                # Negation flips a positive into a negative ("not happy").
                score += -1 if negate else 1
            elif token in self.negative_words:
                # Negation flips a negative into a positive ("not bad").
                score += 1 if negate else -1

            # A negation word affects only the NEXT word, so set the flag
            # for the upcoming token and clear it otherwise.
            negate = token in NEGATION_WORDS

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        # Count how much positive vs. negative signal fired (with negation),
        # so we can tell a true "neutral" (no signal) apart from a "mixed"
        # post where positives and negatives cancel out.
        tokens = self.preprocess(text)
        pos_hits = 0
        neg_hits = 0
        negate = False
        for token in tokens:
            if token in self.positive_words:
                neg_hits += 1 if negate else 0
                pos_hits += 0 if negate else 1
            elif token in self.negative_words:
                pos_hits += 1 if negate else 0
                neg_hits += 0 if negate else 1
            negate = token in NEGATION_WORDS

        score = pos_hits - neg_hits

        # "Mixed" only when both sides fired AND they balance out (no clear
        # winner). If one side outweighs the other, trust the net score so a
        # sarcastic "I love getting stuck in traffic" (love +1, stuck -1,
        # traffic -1 = -1) reads as negative rather than mixed.
        if pos_hits > 0 and neg_hits > 0 and score == 0:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )


if __name__ == "__main__":
    # Quick check that preprocess() behaves as expected.
    from dataset import SAMPLE_POSTS, TRUE_LABELS

    analyzer = MoodAnalyzer()
    correct = 0
    for post, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        predicted = analyzer.predict_label(post)
        match = "✓" if predicted == true_label else "✗"
        if predicted == true_label:
            correct += 1
        print(f"{match} {post!r}")
        print(f"    tokens    -> {analyzer.preprocess(post)}")
        print(f"    score     -> {analyzer.score_text(post)}")
        print(f"    predicted -> {predicted:<8} true -> {true_label}")

    print(f"\nAccuracy: {correct}/{len(SAMPLE_POSTS)} = {correct / len(SAMPLE_POSTS):.0%}")
