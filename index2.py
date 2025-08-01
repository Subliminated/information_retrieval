#%%
import re
import string
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk

#%%
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

lemmatizer = WordNetLemmatizer()
#%%
def normalize_text(text):
    # 1. Case insensitive
    text = text.lower()

    # 2. Handle abbreviations like U.S. -> US
    text = re.sub(r"\b([A-Z])\.(?=[A-Z]\.)", r"\1", text.upper())  # US. -> US
    text = re.sub(r"\.", "", text)  # remove leftover periods in abbreviations

    # 3. Remove possessives like "cat's" → "cat"
    text = re.sub(r"(\'s|s\')", "", text)

    # 4. Replace commas in numbers (1,000 → 1000)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # 5. Remove decimal numbers (e.g., 1.25 → '')
    text = re.sub(r"\b\d+\.\d+\b", "", text)

    # 6. Handle hyphen rules
    def handle_hyphen(match):
        parts = match.group().split("-")
        if len(parts[0]) < 3:
            return match.group()  # keep as is
        else:
            return " ".join(parts)  # split into separate words

    text = re.sub(r"\b[\w]+-[\w]+\b", handle_hyphen, text)

    # 7. Tokenize
    tokens = word_tokenize(text)

    # 8. POS tag
    pos_tags = pos_tag(tokens)

    # 9. Lemmatize (normalize tense, plural/singular)
    lemmatized = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
        if token.isalnum()  # remove punctuation
    ]

    return lemmatized
#%%
doc = "The U.S. co-authors' five-year set-aside breach was in-depth!"
print(normalize_text(doc))
# Output: ['us', 'co-author', 'five', 'year', 'set', 'aside', 'breach', 'be', 'in-depth']


#%%

from collections import defaultdict

def build_inverted_index(docs):
    index = defaultdict(lambda: defaultdict(list))  # term -> doc_id -> [positions]

    for doc_id, text in docs.items():
        terms = normalize_text(text)
        for position, term in enumerate(terms):
            index[term][doc_id].append(position)

    return index
# %%
documents = {
    "doc1": "The U.S. co-authors' five-year set-aside breach was in-depth!",
    "doc2": "Set asides breached by co-authored ex-wives."
}

inverted_index = build_inverted_index(documents)
dict(inverted_index)
# %%
