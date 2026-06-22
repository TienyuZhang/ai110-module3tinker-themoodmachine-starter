"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Common "complaint" words. These often appear in sarcastic posts where a
    # single positive keyword (e.g. "love") would otherwise dominate, such as
    # "I love getting stuck in traffic". Having them here lets the negative
    # context push back instead of the post reading as purely positive.
    "stuck",
    "traffic",
    "cancelled",
    "delayed",
    "waiting",
    "annoyed",
    "nervous",
    "numb",
]

# ---------------------------------------------------------------------
# Emoji / emoticon sentiment
# ---------------------------------------------------------------------
# preprocess() in mood_analyzer.py splits these out as their own tokens, so
# listing them here lets the scorer treat them as real sentiment signals.
# Note: some emojis are genuinely ambiguous (😭 can be joy or grief, 💀 is
# usually "dying laughing"), so they are intentionally left unscored.

POSITIVE_EMOJIS = [
    "🙂", "😊", "😀", "😁", "😄", "😍", "🥰", "❤️", "👍", "🎉", "🔥", "😂", "✨",
    ":)", ":-)", "(:", ":d", "<3",
]

NEGATIVE_EMOJIS = [
    "🙄", "😡", "😠", "😢", "😞", "😔", "👎", "😩", "😫",
    ":(", ":-(", "):", ":/",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "lowkey nervous about the exam but we move 🥲",
    "wow another monday, can't wait 🙄",
    "no cap this playlist is fire 🔥🔥",
    "idk how i feel rn... kinda numb",
    "passed my driving test FINALLY lets gooo 😭🎉",
    "they cancelled the trip lol cool cool cool",
    "just chilling at home, nothing much going on",
    "i hate that i love this song so much 💀",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "lowkey nervous about the exam but we move 🥲" (anxiety + resilience)
    "negative",  # "wow another monday, can't wait 🙄" (sarcasm — really dreading it)
    "positive",  # "no cap this playlist is fire 🔥🔥" (slang praise)
    "neutral",   # "idk how i feel rn... kinda numb" (flat / undecided — edge case)
    "positive",  # "passed my driving test FINALLY lets gooo 😭🎉" (happy-crying emoji)
    "negative",  # "they cancelled the trip lol cool cool cool" (sarcasm masking disappointment)
    "neutral",   # "just chilling at home, nothing much going on"
    "mixed",     # "i hate that i love this song so much 💀" (love + mock-frustration — edge case)
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
