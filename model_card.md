# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I built, tuned, and compared **both** models on the same labeled dataset.

## 1. Model Overview

**Model type:**
I used and compared **both** models — the hand-written rule based model in `mood_analyzer.py` and the scikit-learn ML model (CountVectorizer + LogisticRegression) in `ml_experiments.py` — trained and evaluated on the same `SAMPLE_POSTS` / `TRUE_LABELS`.

**Intended purpose:**
Classify short, social-media-style text messages into one of four moods: **positive, negative, neutral, or mixed**.

**How it works (brief):**
- *Rule based:* the text is preprocessed (lowercased, punctuation stripped, emojis split into their own tokens, elongated words like "gooo" → "goo" normalized). Each token is scored `+1` if it's a known positive word/emoji and `−1` if negative. A negation word ("not", "never", "can't"…) flips the sign of the next sentiment token. The net score and the mix of signals are mapped to a label.
- *ML:* `CountVectorizer` turns each post into a bag-of-words count vector, and `LogisticRegression` learns word weights from the labeled examples. No rules are written by hand — the patterns are learned from `TRUE_LABELS`.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains **14 posts** (6 starter posts plus 8 I added). I wrote the new posts to deliberately include realistic, hard-to-classify language: slang ("lowkey", "no cap", "idk"), emojis (🥲 🙄 🔥 😭 🎉 💀), sarcasm, and bittersweet/mixed feelings. Label balance is roughly even: 4 positive, 4 negative, 3 mixed, 3 neutral.

**Labeling process:**
I labeled each post by the *overall* feeling a human reader would take away, not just the literal words. For sarcasm I labeled by intended meaning (e.g. "wow another monday, can't wait 🙄" → negative). For posts with both positive and negative cues that genuinely balance out, I used "mixed".

Posts that were hard to label / could reasonably get more than one label:
- `"idk how i feel rn... kinda numb"` — I called it **neutral**, but **negative** is defensible.
- `"i hate that i love this song so much 💀"` — **mixed** (literal "hate" + "love"), but arguably positive in spirit.
- `"passed my driving test FINALLY lets gooo 😭🎉"` — **positive**, even though 😭 looks negative out of context.

**Important characteristics of your dataset:**
- Contains slang and emojis
- Includes sarcasm ("can't wait 🙄", "cool cool cool")
- Several posts express genuinely mixed feelings
- Many posts are short and ambiguous

**Possible issues with the dataset:**
- **Very small** (14 posts) — far too few to train a reliable ML model.
- **Ambiguity** — several posts have no single "correct" label, so even the ground truth is debatable.
- **Coverage gaps** — limited vocabulary; many real-world words/emojis never appear, so the ML model has never "seen" them and the rule based model has no entry for them.
- Written by one person, so it reflects one labeling perspective and one slang vocabulary.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- **Word lists:** positive words score `+1`, negative words score `−1`. I expanded `NEGATIVE_WORDS` with common "complaint" words (`stuck`, `traffic`, `cancelled`, `nervous`, `numb`…) so a single positive keyword can't dominate a clearly negative sentence.
- **Negation:** a negation word (`not`, `never`, `can't`, `don't`…) flips the sign of the following sentiment token, so "not happy" scores `−1` and "not bad" scores `+1`.
- **Emoji handling:** `preprocess` splits emojis into separate tokens, and they're scored as real signals via `POSITIVE_EMOJIS` / `NEGATIVE_EMOJIS` (🙂 🔥 🎉 → positive; 🙄 😡 😞 → negative). Genuinely ambiguous emojis (😭, 💀, 🥲) are intentionally **left unscored**.
- **Label thresholds:** `score > 0` → positive, `score < 0` → negative, `score == 0` with no signals → neutral. **"Mixed" only when both positive and negative signals fired AND they balance to a net score of 0** — so sarcastic "I love getting stuck in traffic" (net −1) reads negative, not mixed.

**Strengths of this approach:**
- Fully **transparent** — every prediction can be explained word by word (`explain()`).
- Handles **negation** and **emojis** that a plain bag-of-words ignores.
- **Generalizes** to unseen text: any sentence using known words/emojis is scored sensibly, even if it never appeared in the dataset.

**Weaknesses of this approach:**
- Cannot detect **sarcasm** unless the sentence happens to name something negative.
- Blind to any word/emoji **not in my lists**.
- Misses **subtle/implied** moods with no explicit sentiment word (e.g. "tired but hopeful" — "hopeful" isn't in the vocabulary).
- **Accuracy: 11/14 (79%)** on the dataset.

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words via `CountVectorizer` — each post becomes a vector of word counts. **Word order is discarded.**

**Training data:**
Trained on the same `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py` (14 posts).

**Training behavior:**
With so few examples, the model effectively **memorizes** the training set. Adding or relabeling posts directly changes which words map to which label. Because the vocabulary is tiny, single distinctive words ("terrible", "excited") become near-perfect predictors *for the training data*.

**Strengths and weaknesses:**
- *Strengths:* learns word→label associations automatically with no hand-written rules; picked up emoji and slang cues (🔥, 🙄) on its own because they appeared in training.
- *Weaknesses:* severe **overfitting** — 100% training accuracy is misleading. It has no concept of negation (order is dropped), and any word it never saw in training is invisible, so it fails on genuinely new sentences in the interactive loop.

## 5. Evaluation

**How you evaluated the model:**
Both models were evaluated on the 14 labeled posts in `dataset.py`.
- **Rule based: 11/14 = 79%**
- **ML model: 14/14 = 100%** — but this is **training accuracy** (trained and tested on the *same* data), so it overstates real performance.

**Examples of correct predictions:**
- `"Today was a terrible day"` → both **negative**. "terrible" is an unambiguous negative cue.
- `"no cap this playlist is fire 🔥🔥"` → both **positive**. The rule based model scored the 🔥 emojis; the ML model learned "fire"/🔥 from training.
- `"I am not happy about this"` → both **negative**. The rule based model's negation flip turns "happy" negative; the ML model learned the whole phrase's label.

**Examples of incorrect predictions:**
Rule based model's 3 misses (the ML model got all 3 "right" only because it memorized them):
- `"Feeling tired but kind of hopeful"` → predicted **negative**, true **mixed**. "tired" is in the list but "hopeful" is not, so only the negative side registered.
- `"lowkey nervous about the exam but we move 🥲"` → predicted **negative**, true **mixed**. "nervous" scored negative; the hopeful "we move 🥲" half has no scored tokens (🥲 is intentionally unscored).
- `"idk how i feel rn... kinda numb"` → predicted **negative**, true **neutral**. "numb" is in `NEGATIVE_WORDS`, but here it reads as flat/emotionless.

**How their failures differ:**
The rule based model fails on **words it doesn't know or moods that are implied**. The ML model shows *no* failures here only because it was tested on its own training data — its real failures appear on **brand-new sentences** in the interactive loop, where any unseen word is ignored and negation is lost.

### Rule based vs. learned ML model — short comparison

**Did the learned model behave differently?**
Yes. On the 14-post dataset the ML model scored 14/14 vs. the rule based model's 11/14, but that gap is misleading — the ML model was tested on the same posts it trained on, so it *memorized* them rather than reasoning about them. The rule based model applies the same explicit logic to every input; the ML model just maps known words to whichever label they co-occurred with in training.

**Did it fix failures or introduce new ones?**
On paper it "fixed" the rule based model's 3 misses (`tired but hopeful`, `nervous … 🥲`, `kinda numb`) — but only because it had already seen those exact posts and their labels, not because it understands them. It **introduces a new, hidden failure mode**: any sentence containing a word it never saw in training is effectively invisible, and because it ignores word order it cannot represent negation at all (`"not happy"` is just the tokens "not" + "happy"). So it trades the rule based model's *visible, explainable* errors for *silent* ones on new text.

**How sensitive was it to the labels I created?**
Very. I tested this directly: flipping a **single** label (`"So excited for the weekend"` from positive → negative) changed the model's prediction on a brand-new sentence `"I am so excited"` from **positive to negative**. With only 14 examples, each label carries enormous weight, so the model is fragile and easily skewed by one labeling choice or mistake — whereas the rule based model's behavior changes only when I deliberately edit its word lists or rules.

## 6. Limitations

- The dataset is **tiny (14 posts)** — far too small for a trustworthy ML model, and the ML accuracy is training accuracy, not a real generalization estimate.
- Neither model handles **sarcasm** reliably.
- Performance depends heavily on **the words and labels I chose** — change the vocabulary and the results shift.
- Designed for **short** posts; behavior on longer or multi-sentence text is untested.
- The ML model **ignores word order**, so it cannot represent negation at all.

## 7. Bias and Scope

**Who this model is optimized for:**
The dataset is short, informal, **English** social-media text written by **one young/student author**. The vocabulary and emoji choices reflect current internet slang ("lowkey", "no cap", "idk", "lets gooo", 🔥 🙄 🥲). So the model works best for people who write the way the author does: **casual, English-speaking, emoji-using, broadly Gen-Z online culture.**

**Who it might misinterpret:**
- **Other dialects and communities** — AAVE, regional slang, or in-group terms aren't in the word lists (rule based) and never appeared in training (ML), so their sentiment is missed or guessed.
- **Non-English or code-switched text** — essentially unscored; out of scope entirely.
- **Different generations / formal writers** — someone who writes politely or formally ("I am rather disappointed") gives few of the slang/emoji cues the model leans on.
- **Cultural emoji differences** — emoji meaning varies by culture and age; 🙂 can read as passive-aggressive and 💀 as morbid to readers who don't share the "dying laughing" convention, yet the model bakes in one interpretation.
- **Sarcasm-heavy or understated styles** — already a weak spot, and disproportionately affects communities whose humor relies on it.

**Scope statement:**
This is a small **classroom/demo** model for short, casual English posts. It is **not** validated for, and should not be used on, other languages, formal text, longer documents, or any high-stakes decision about a real person.

## 8. Ethical Considerations

- **Misclassifying distress:** a post expressing genuine sadness or a cry for help could be flagged "neutral" or "mixed", which is dangerous if such a system gated mental-health support or moderation.
- **Bias across communities:** the vocabulary and slang reflect one person's usage. Dialects, AAVE, non-English phrases, or other slang would be systematically misread, disadvantaging some groups.
- **Privacy:** analyzing personal messages for mood is sensitive surveillance; it should require consent, clear purpose, and data minimization.
- **Overconfidence:** the ML model's 100% accuracy could mislead someone into trusting it far more than its tiny, overfit training set justifies.

## 9. Ideas for Improvement

- **Add much more labeled data**, and split into separate **train / test sets** so accuracy reflects generalization, not memorization.
- Use **TF-IDF** instead of raw counts, and/or **n-grams** so the ML model can capture short phrases like "not happy".
- **Richer preprocessing** for emojis and slang (e.g. a sentiment lexicon, handling repeated/elongated words as intensity).
- **Improve the rule based scoring** — weighted words ("hate" stronger than "annoyed"), intensity from elongation/CAPS, and more emoji coverage.
- Try a **small neural network or pre-trained transformer** that understands context and word order (and therefore negation and some sarcasm).
- Build a true **held-out evaluation set** with multiple human labelers to measure agreement on ambiguous posts.
