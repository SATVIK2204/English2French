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

    def __remove(self, text, cnt, frequent, n_freq, rare, n_rare):

        FREQWORDS = set([w for (w, wc) in cnt.most_common(n_freq)])
        RAREWORDS = set([w for (w, wc) in cnt.most_common()[: -n_rare - 1 : -1]])
        if frequent:
            text = " ".join(
                [word for word in str(text).split() if word not in FREQWORDS]
            )
        if rare:
            text = " ".join(
                [word for word in str(text).split() if word not in RAREWORDS]
            )

        return text

    # Remove the most frequent words
    def remove_frequent_rare(
        self, lines, frequent=False, n_freq=10, rare=False, n_rare=10
    ):
        cnt = Counter()
        for line in lines:
            for word in line.split():
                cnt[word] += 1
        length=len(lines)
        i=0
        new_line = []
        for line in lines:
            new_line.append(self.__remove(line, cnt, frequent, n_freq, rare, n_rare))
            i = i + 1
            if i % 1000 == 0:
                print(f"{i} examples cleaned out of {length}")
            if i==length:
                print('Cleaning Done')


        return new_line

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
        length = len(lines)
        i = 0
        for line in lines:
            
            cleaned_lines.append(
                self.__clean_text(line, remove_stopwords, stem, lemmitize)
            )
            i = i + 1
            if i % 10000 == 0:
                print(f"{i} examples cleaned out of {length}")
            if i==length:
                print('Cleaning Done')

        return cleaned_lines
