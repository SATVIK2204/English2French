import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from collections import Counter


class DfCleaner:
    def __init__(self):
        pass

    # Remove the most frequent words
    def remove_frequent_rare(self, lines, frequent=False, n_freq=10, rare=False, n_rare=10):
        cnt = Counter()
        for line in lines:
            for word in line.split():
                cnt[word] += 1
        if frequent:
            FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
            for line in lines:
                

    # Performs Various operations to clean the text

    def __clean_text(self, text, remove_stopwords, stem, lemmitize):
        text = text.lower()

        # Remove numbers
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"  ", " ", text)
        text = re.sub(r" — ", " ", text)
        # Remove unwanted punctuations
        text = re.sub(r'[_"”\-;%()’“|‘+&=*%.!,?:#$@\[\]/]', " ", text)
        text = re.sub(r"\'", " ", text)
        # Remove leading and trailing whitespaces
        text = re.sub("\s+", " ", text).strip()

        # Remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        # Stemming the text
        if stem:
            stemmer = PorterStemmer()
            text = text.split()
            text = [stemmer.stem(word) for word in text.split()]
            text = " ".join(text)

        # Lemmatizing the text
        if lemmitize:
            lemmatizer = WordNetLemmatizer()
            # Here we will use POS-tagging to pass on to lemmatizer using wornet to get the correct lemmitization
            wordnet_map = {
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "J": wordnet.ADJ,
                "R": wordnet.ADV,
            }
            pos_tagged_text = nltk.pos_tag(text.split())
            text = [
                lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                for word, pos in pos_tagged_text
            ]
            text = " ".join(text)

        return text

    def clean(self, lines, remove_stopwords=True, stem=True, lemmitize=True):
        cleaned_lines = []
        for line in lines:
            cleaned_lines.append(
                self.__clean_text(line, remove_stopwords, stem, lemmitize)
            )

        return cleaned_lines
